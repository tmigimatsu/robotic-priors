#include "CSaiEnv.h"

#include <iostream>
#include <fstream>
#include <cmath>

#include <signal.h>
static volatile bool g_runloop = true;
void stop(int) { g_runloop = false; }

// Return true if any elements in the Eigen::MatrixXd are NaN
template<typename Derived>
static inline bool isnan(const Eigen::MatrixBase<Derived>& x) {
	return (x.array() != x.array()).any();
}

static void glfwError(int error, const char* description) {
	std::cerr << "GLFW Error: " << description << std::endl;
	exit(1);
}

/**
 * CSaiEnv::reset()
 * ----------------
 * Reset robot simulation.
 */
void CSaiEnv::reset() {
	controller_counter_ = 0;
	command_torques_.setZero();

	// Home configuration for Kuka iiwa
	// q_des_ << 90, -30, 0, 60, 0, -90, -60;
	// q_des_ *= M_PI / 180.0;
	q_des_ << 1.570796, -0.151874, 0, 1.681582, 0, -1.308137, -1.570796;
	dq_des_.setZero();

	// Desired end effector position
	x_des_ << 0, -0.45, 0.55;
	action_ << 0, 0, 0;
	dx_des_.setZero();
	R_des_ << 1,  0,  0,
		      0, -1,  0,
		      0,  0, -1;

	// Reset model
	sim->setJointPositions(kRobotName, q_des_);
	sim->setJointVelocities(kRobotName, dq_des_);
	try {
		readRedisValues();
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	updateModel();
}

/**
 * CSaiEnv::reset()
 * ----------------
 * Reset for OpenAI gym.
 */
void CSaiEnv::reset(uint8_t *observation) {
	reset();

	gl_buffer_ = observation;
	syncGraphics();
};

/**
 * CSaiEnv::step()
 * ---------------
 * Step dynamics for OpenAI gym.
 */
bool CSaiEnv::step(const double *action, uint8_t *observation, double& reward, double *info) {
	for (int i = 0; i < kControlFreq / kEnvironmentFreq; i++) {
		++controller_counter_;
		action_ << action[0], action[1], 0;
		computeOperationalSpaceControlTorques();
		writeRedisValues();
		try {
			readRedisValues();
		} catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
		updateModel();
	}

	gl_buffer_ = observation;
	// Tell graphics to update
	mutex_graphics_.lock();
	update_graphics_ = true;
	mutex_graphics_.unlock();
	cv_.notify_all();

	// Compute reward
	if (((x_ - Eigen::Vector3d(0, -0.45, 0.55)).array().abs() > 0.2 - kWallTolerance).any()) {
		reward = -1;
	} else if ((x_ - Eigen::Vector3d(-0.2, -0.25, 0.55)).norm() < kCornerDistance) {
		reward = 10;
	} else {
		reward = 0;
	}

	// Wait for graphics to finish
	std::unique_lock<std::mutex> lock_graphics(mutex_graphics_);
	cv_.wait(lock_graphics, [this]{
		return !update_graphics_;
	});

	// Finish episode after 10s
	bool done = controller_counter_ / kControlFreq >= 10 ||
	            ((x_ - Eigen::Vector3d(0, -0.45, 0.55)).array().abs() > 0.2).any();

	// Return debug info
	if (info != nullptr) {
		for (int i = 0; i < 3; i++) info[  i] = x_(i);
		for (int i = 0; i < 3; i++) info[3+i] = dx_(i);
	}

	return done;
};

/**
 * CSaiEnv::readRedisValues()
 * --------------------------
 * Retrieve all read keys from Redis.
 */
void CSaiEnv::readRedisValues() {
	// Read from Redis current sensor values
	sim->getJointPositions(kRobotName, robot_->_q);
	sim->getJointVelocities(kRobotName, robot_->_dq);
}

/**
 * CSaiEnv::writeRedisValues()
 * ---------------------------
 * Send all write keys to Redis.
 */
void CSaiEnv::writeRedisValues() {
	sim->setJointTorques(kRobotName, command_torques_);
	for (int i = 0; i < kSimulationFreq / kControlFreq; i++) {
		sim->integrate(1.0 / kSimulationFreq);
	}
	redis_.setEigenMatrix(KEY_JOINT_POSITIONS, robot_->_q);
	redis_.setEigenMatrix(KEY_EE_POS, x_);
	redis_.setEigenMatrix(KEY_EE_POS_DES, action_);
}

/**
 * CSaiEnv::updateModel()
 * ----------------------
 * Update the robot model and all the relevant member variables.
 */
void CSaiEnv::updateModel() {
	std::lock_guard<std::mutex> lock(mutex_robot_);

	// Update the model
	robot_->updateModel();

	// Forward kinematics
	x_ = robot_->position("link6", Eigen::Vector3d::Zero());
	dx_ = robot_->linearVelocity("link6", Eigen::Vector3d::Zero());
	w_ = robot_->angularVelocity("link6");
	R_ee_to_base_ = robot_->rotation("link6");

	// Jacobians
	J_ = robot_->J("link6", Eigen::Vector3d::Zero());
	N_ = robot_->nullspaceMatrix(J_);

	// Dynamics
	Lambda_ = robot_->taskInertiaMatrixWithPseudoInv(J_);
	g_ = robot_->gravityVector();
}

/**
 * CSaiEnv::computeJointSpaceControlTorques()
 * ------------------------------------------
 * Controller to initialize robot to desired joint position.
 */
CSaiEnv::ControllerStatus CSaiEnv::computeJointSpaceControlTorques() {
	// Finish if the robot has converged to q_initial
	Eigen::VectorXd q_err = robot_->_q - q_des_;
	Eigen::VectorXd dq_err = robot_->_dq - dq_des_;
	if (q_err.norm() < kToleranceInitQ && dq_err.norm() < kToleranceInitDq) {
		return FINISHED;
	}

	// Compute torques
	Eigen::VectorXd ddq = -kp_joint_ * q_err - kv_joint_ * dq_err;
	command_torques_ = robot_->_M * ddq + g_;
	return RUNNING;
}

/**
 * CSaiEnv::computeOperationalSpaceControlTorques()
 * ------------------------------------------------
 * Controller to move end effector to desired position.
 */
CSaiEnv::ControllerStatus CSaiEnv::computeOperationalSpaceControlTorques() {
	// PD position control with velocity saturation
	Eigen::Vector3d x_err = x_ - x_des_;
	x_err(0) = kp_action_ / kp_pos_ * (action_(0));
	x_err(1) = kp_action_ / kp_pos_ * (action_(1));
	Eigen::Vector3d dx_err = dx_ - dx_des_;
	Eigen::Vector3d ddx = -kp_pos_ * x_err - kv_pos_ * dx_err;
	// dx_des_ = -(kp_pos_ / kv_pos_) * x_err;
	// double v = kMaxVelocity / dx_des_.norm();
	// if (v > 1) v = 1;
	// Eigen::Vector3d dx_err = dx_ - v * dx_des_;
	// Eigen::Vector3d ddx = -kv_pos_ * dx_err;

	// Orientation
	Eigen::Vector3d dPhi = robot_->orientationError(R_des_, R_ee_to_base_);
	Eigen::Vector3d dw = -kp_ori_ * dPhi - kv_ori_ * w_;

	// Nullspace posture control and damping
	Eigen::VectorXd q_err = robot_->_q - q_des_;
	Eigen::VectorXd dq_err = robot_->_dq - dq_des_;
	Eigen::VectorXd ddq = -kp_joint_ * q_err - kv_joint_ * dq_err;

	// Control torques
	Eigen::VectorXd ddx_dw(6);
	ddx_dw << ddx, dw;
	Eigen::VectorXd F = Lambda_ * ddx_dw;
	Eigen::VectorXd F_posture = robot_->_M * ddq;
	command_torques_ = J_.transpose() * F + N_.transpose() * F_posture + g_;

	return RUNNING;
}

/**
 * public CSaiEnv::initialize()
 * ----------------------------
 * Initialize timer and Redis client
 */
void CSaiEnv::initialize() {
	// Create a loop timer
	timer_.setLoopFrequency(kControlFreq);   // 1 KHz
	// timer.setThreadHighPriority();  // make timing more accurate. requires running executable as sudo.
	timer_.setCtrlCHandler(stop);    // exit while loop on ctrl-c
	timer_.initializeTimer(kInitializationPause); // 1 ms pause before starting loop

	// Start redis client
	// Make sure redis-server is running at localhost with default port 6379
	redis_.connect(kRedisHostname, kRedisPort);
}

/**
 * public CSaiEnv::syncGraphics()
 * -------------------------------
 * Wait for graphics to finish update.
 */
void CSaiEnv::syncGraphics() {
	// Tell graphics to update
	mutex_graphics_.lock();
	update_graphics_ = true;
	mutex_graphics_.unlock();
	cv_.notify_all();

	// Wait for graphics to finish
	std::unique_lock<std::mutex> lock_graphics(mutex_graphics_);
	cv_.wait(lock_graphics, [this]{
		return !update_graphics_;
	});
}

/**
 * public CSaiEnv::graphicsMain()
 * ------------------------------
 * Graphics thread.
 */
void CSaiEnv::graphicsMain(std::shared_ptr<Graphics::GraphicsInterface> graphics) {
	Graphics::ChaiGraphics *chai = dynamic_cast<Graphics::ChaiGraphics *>(graphics->_graphics_internal);

	// Start visualization
    glfwSetErrorCallback(glfwError);
    glfwInit();

	// Create window
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwWindowHint(GLFW_VISIBLE, 0);
    GLFWwindow *window = glfwCreateWindow(kWindowWidth, kWindowHeight, "SAI2 OpenAI Gym Environment", nullptr, nullptr);
	glfwSetWindowPos(window, 0.5 * (mode->height - kWindowHeight), 0.5 * (mode->height - kWindowHeight));
	glfwShowWindow(window);
    glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	chai->setCameraPose(kCameraName, kCameraPos, kCameraVertical, kCameraLookat);

	while (g_runloop && !glfwWindowShouldClose(window)) {
		// Wait for update notification
		std::unique_lock<std::mutex> lock_graphics(mutex_graphics_);
		cv_.wait(lock_graphics, [this]{
			return update_graphics_ || !g_runloop;
		});
		if (!g_runloop) break;

		// Update robot visualization
		mutex_robot_.lock();
		chai->updateGraphics(kRobotName, robot_.get());
		mutex_robot_.unlock();

		// Render graphics
		chai->render(kCameraName, kWindowWidth, kWindowHeight);
		glfwSwapBuffers(window);
		glFinish();

		// Take screenshot
		if (gl_buffer_ == nullptr) {
			gl_buffer_ = new uint8_t[3 * kWindowWidth * kWindowHeight];
		}
		glReadPixels(0, 0, kWindowWidth, kWindowHeight, GL_RGB, GL_UNSIGNED_BYTE, gl_buffer_);
		if (glGetError() != GL_NO_ERROR) g_runloop = false;

		// Notify threads waiting on graphics
		update_graphics_ = false;
		lock_graphics.unlock();
		cv_.notify_all();
	}
	g_runloop = false;

	glfwDestroyWindow(window);
	glfwTerminate();
}


/**
 * public CSaiEnv::runLoop()
 * -------------------------
 * CSaiEnv state machine
 */
void CSaiEnv::runLoop() {

	while (g_runloop) {
		// Wait for next scheduled loop (controller must run at precise rate)
		// timer_.waitForNextLoop();
		++controller_counter_;

		// Get latest sensor values from Redis and update robot model
		try {
			readRedisValues();
		} catch (std::exception& e) {
			if (controller_state_ != REDIS_SYNCHRONIZATION) {
				std::cout << e.what() << " Aborting..." << std::endl;
				break;
			}
			std::cout << e.what() << " Waiting..." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));
			continue;
		}
		updateModel();

		switch (controller_state_) {
			// Wait until valid sensor values have been published to Redis
			case REDIS_SYNCHRONIZATION:
				if (isnan(robot_->_q)) continue;
				std::cout << "Redis synchronized. Switching to joint space controller." << std::endl;
				controller_state_ = JOINT_SPACE_INITIALIZATION;
				break;

			// Initialize robot to default joint configuration
			case JOINT_SPACE_INITIALIZATION:
				if (computeJointSpaceControlTorques() == FINISHED) {
					std::cout << "Joint position initialized. Switching to operational space controller." << std::endl;
					controller_state_ = CSaiEnv::OP_SPACE_POSITION_CONTROL;
				};
				break;

			// Control end effector to desired position
			case OP_SPACE_POSITION_CONTROL:
				computeOperationalSpaceControlTorques();
				break;

			// Invalid state. Zero torques and exit program.
			default:
				std::cout << "Invalid controller state. Stopping controller." << std::endl;
				g_runloop = false;
				command_torques_.setZero();
				break;
		}

		// Check command torques before sending them
		if (isnan(command_torques_)) {
			std::cout << "NaN command torques. Sending zero torques to robot." << std::endl;
			command_torques_.setZero();
		}

		// Send command torques
		writeRedisValues();

		if (controller_counter_ % (kControlFreq / kEnvironmentFreq) == 0) {
			mutex_graphics_.lock();
			update_graphics_ = true;
			mutex_graphics_.unlock();
			cv_.notify_all();
		}
	}

	// Zero out torques before quitting
	command_torques_.setZero();
	// redis_.setEigenMatrix(KEY_COMMAND_TORQUES, command_torques_);

	thread_graphics.join();
}

int main(int argc, char** argv) {

	// Parse command line
	if (argc != 4) {
		std::cout << "Usage: demo_app <path-to-world.urdf> <path-to-robot.urdf> <robot-name>" << std::endl;
		exit(0);
	}
	// Argument 0: executable name
	// Argument 1: <path-to-world.urdf>
	std::string world_file(argv[1]);
	// Argument 2: <path-to-robot.urdf>
	std::string robot_file(argv[2]);
	// Argument 3: <robot_-name>
	std::string robot_name(argv[3]);

	// Set up signal handler
	signal(SIGABRT, &stop);
	signal(SIGTERM, &stop);
	signal(SIGINT, &stop);

	// Load robot
	std::cout << "Loading robot: " << robot_file << std::endl;
	auto robot = std::make_shared<Model::ModelInterface>(robot_file, Model::rbdl, Model::urdf, false);
	auto sim = std::make_shared<Simulation::SimulationInterface>(world_file, Simulation::sai2simulation, Simulation::urdf, false);
	auto graphics = std::make_shared<Graphics::GraphicsInterface>(world_file, Graphics::chai, Graphics::urdf, true);

	// Start controller app
	std::cout << "Initializing app with " << robot_name << std::endl;
	CSaiEnv app(std::move(robot), std::move(sim), std::move(graphics), robot_name);
	app.initialize();
	std::cout << "App initialized. Waiting for Redis synchronization." << std::endl;
	app.runLoop();

	return 0;
}


