#include "SaiGym.h"

#include <iostream>
#include <fstream>

#include <signal.h>
static volatile bool g_runloop = true;
void stop(int) { g_runloop = false; }

// Return true if any elements in the Eigen::MatrixXd are NaN
template<typename Derived>
static inline bool isnan(const Eigen::MatrixBase<Derived>& x) {
	return (x.array() != x.array()).any();
}

using namespace std;

static void glfwError(int error, const char* description) {
	cerr << "GLFW Error: " << description << endl;
	exit(1);
}

/**
 * SaiGym::readRedisValues()
 * ------------------------------
 * Retrieve all read keys from Redis.
 */
void SaiGym::readRedisValues() {
	// Read from Redis current sensor values
	sim->getJointPositions(kRobotName, robot->_q);
	sim->getJointVelocities(kRobotName, robot->_dq);
}

void SaiGym::publishEnvironment() {
	graphics->updateGraphics(kRobotName, robot.get());
	graphics->render(kCameraName, kWindowWidth, kWindowHeight);
	glfwSwapBuffers(window_);
	glFinish();

	glReadPixels(0, 0, kWindowWidth, kWindowHeight, GL_RGB, GL_UNSIGNED_BYTE, gl_buffer_);
	for (int y = 0; y < kWindowHeight; y++) {
		for (int x = 0; x < kWindowWidth; x++) {
			buffer_r_(y, x) = static_cast<int>(gl_buffer_[3 * (kWindowWidth * y + x) + 0]);
			buffer_g_(y, x) = static_cast<int>(gl_buffer_[3 * (kWindowWidth * y + x) + 1]);
			buffer_b_(y, x) = static_cast<int>(gl_buffer_[3 * (kWindowWidth * y + x) + 2]);
		}
	}

	if (glGetError() != GL_NO_ERROR) g_runloop = false;
}

/**
 * SaiGym::writeRedisValues()
 * -------------------------------
 * Send all write keys to Redis.
 */
void SaiGym::writeRedisValues() {
	sim->setJointTorques(kRobotName, command_torques_);
	for (int i = 0; i < kSimulationFreq / kControlFreq; i++) {
		sim->integrate(1.0 / kSimulationFreq);
	}
	redis_.setEigenMatrix(KEY_JOINT_POSITIONS, robot->_q);
}

/**
 * SaiGym::updateModel()
 * --------------------------
 * Update the robot model and all the relevant member variables.
 */
void SaiGym::updateModel() {
	// Update the model
	robot->updateModel();

	// Forward kinematics
	robot->position(x_, "link6", Eigen::Vector3d::Zero());
	robot->linearVelocity(dx_, "link6", Eigen::Vector3d::Zero());

	// Jacobians
	robot->Jv(Jv_, "link6", Eigen::Vector3d::Zero());
	N_ = robot->nullspaceMatrix(Jv_);
	Eigen::MatrixXd Jw = robot->Jw("link6") * N_;
	Eigen::MatrixXd Nw = robot->nullspaceMatrix(Jw, N_);

	// Dynamics
	robot->taskInertiaMatrixWithPseudoInv(Lambda_x_, Jv_);
	robot->gravityVector(g_);
}

/**
 * SaiGym::computeJointSpaceControlTorques()
 * ----------------------------------------------
 * Controller to initialize robot to desired joint position.
 */
SaiGym::ControllerStatus SaiGym::computeJointSpaceControlTorques() {
	// Finish if the robot has converged to q_initial
	Eigen::VectorXd q_err = robot->_q - q_des_;
	Eigen::VectorXd dq_err = robot->_dq - dq_des_;
	if (q_err.norm() < kToleranceInitQ && dq_err.norm() < kToleranceInitDq) {
		return FINISHED;
	}

	// Compute torques
	Eigen::VectorXd ddq = -kp_joint_ * q_err - kv_joint_ * dq_err;
	command_torques_ = robot->_M * ddq + g_;
	return RUNNING;
}

/**
 * SaiGym::computeOperationalSpaceControlTorques()
 * ----------------------------------------------------
 * Controller to move end effector to desired position.
 */
SaiGym::ControllerStatus SaiGym::computeOperationalSpaceControlTorques() {
	// PD position control with velocity saturation
	Eigen::Vector3d x_err = x_ - x_des_;
	// Eigen::Vector3d dx_err = dx_ - dx_des_;
	// Eigen::Vector3d ddx = -kp_pos_ * x_err - kv_pos_ * dx_err_;
	dx_des_ = -(kp_pos_ / kv_pos_) * x_err;
	double v = kMaxVelocity / dx_des_.norm();
	if (v > 1) v = 1;
	Eigen::Vector3d dx_err = dx_ - v * dx_des_;
	Eigen::Vector3d ddx = -kv_pos_ * dx_err;

	// Nullspace posture control and damping
	Eigen::VectorXd q_err = robot->_q - q_des_;
	Eigen::VectorXd dq_err = robot->_dq - dq_des_;
	Eigen::VectorXd ddq = -kp_joint_ * q_err - kv_joint_ * dq_err;

	// Control torques
	Eigen::Vector3d F_x = Lambda_x_ * ddx;
	Eigen::VectorXd F_posture = robot->_M * ddq;
	command_torques_ = Jv_.transpose() * F_x + N_.transpose() * F_posture + g_;

	return RUNNING;
}

/**
 * public SaiGym::initialize()
 * --------------------------------
 * Initialize timer and Redis client
 */
void SaiGym::initialize() {
	// Create a loop timer
	timer_.setLoopFrequency(kControlFreq);   // 1 KHz
	// timer.setThreadHighPriority();  // make timing more accurate. requires running executable as sudo.
	timer_.setCtrlCHandler(stop);    // exit while loop on ctrl-c
	timer_.initializeTimer(kInitializationPause); // 1 ms pause before starting loop

	// Start redis client
	// Make sure redis-server is running at localhost with default port 6379
	redis_.connect(kRedisHostname, kRedisPort);

	// Start visualization
	graphics->getCameraPose(kCameraName, camera_pos_, camera_vertical_, camera_lookat_);
    glfwSetErrorCallback(glfwError);
    glfwInit();

	// Create window
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwWindowHint(GLFW_VISIBLE, 0);
    window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "SAI2 OpenAI Gym Environment", nullptr, nullptr);
	glfwSetWindowPos(window_, 0.5 * (mode->height - kWindowHeight), 0.5 * (mode->height - kWindowHeight));
	glfwShowWindow(window_);
    glfwMakeContextCurrent(window_);
	glfwSwapInterval(1);
	graphics->setCameraPose(kCameraName, camera_pos_, camera_vertical_, camera_lookat_);
}

/**
 * public SaiGym::runLoop()
 * -----------------------------
 * SaiGym state machine
 */
void SaiGym::runLoop() {

	while (!glfwWindowShouldClose(window_) && g_runloop) {
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
				if (isnan(robot->_q)) continue;
				cout << "Redis synchronized. Switching to joint space controller." << endl;
				controller_state_ = JOINT_SPACE_INITIALIZATION;
				break;

			// Initialize robot to default joint configuration
			case JOINT_SPACE_INITIALIZATION:
				if (computeJointSpaceControlTorques() == FINISHED) {
					cout << "Joint position initialized. Switching to operational space controller." << endl;
					controller_state_ = SaiGym::OP_SPACE_POSITION_CONTROL;
				};
				break;

			// Control end effector to desired position
			case OP_SPACE_POSITION_CONTROL:
				computeOperationalSpaceControlTorques();
				break;

			// Invalid state. Zero torques and exit program.
			default:
				cout << "Invalid controller state. Stopping controller." << endl;
				g_runloop = false;
				command_torques_.setZero();
				break;
		}

		// Check command torques before sending them
		if (isnan(command_torques_)) {
			cout << "NaN command torques. Sending zero torques to robot." << endl;
			command_torques_.setZero();
		}

		// Send command torques
		writeRedisValues();

		if (controller_counter_ % (kControlFreq / kEnvironmentFreq) == 0) {
			publishEnvironment();
		}
	}

	// Zero out torques before quitting
	command_torques_.setZero();
	redis_.setEigenMatrix(KEY_COMMAND_TORQUES, command_torques_);

	glfwDestroyWindow(window_);
	glfwTerminate();
}

int main(int argc, char** argv) {

	// Parse command line
	if (argc != 4) {
		cout << "Usage: demo_app <path-to-world.urdf> <path-to-robot.urdf> <robot-name>" << endl;
		exit(0);
	}
	// Argument 0: executable name
	// Argument 1: <path-to-world.urdf>
	string world_file(argv[1]);
	// Argument 2: <path-to-robot.urdf>
	string robot_file(argv[2]);
	// Argument 3: <robot-name>
	string robot_name(argv[3]);

	// Set up signal handler
	signal(SIGABRT, &stop);
	signal(SIGTERM, &stop);
	signal(SIGINT, &stop);

	// Load robot
	cout << "Loading robot: " << robot_file << endl;
	auto robot = make_shared<Model::ModelInterface>(robot_file, Model::rbdl, Model::urdf, false);
	auto sim = std::make_shared<Simulation::SimulationInterface>(world_file, Simulation::sai2simulation, Simulation::urdf, false);
	auto graphics = std::make_shared<Graphics::GraphicsInterface>(world_file, Graphics::chai, Graphics::urdf, true);
	robot->updateModel();

	// Start controller app
	cout << "Initializing app with " << robot_name << endl;
	SaiGym app(move(robot), robot_name, move(sim), graphics);
	app.initialize();
	cout << "App initialized. Waiting for Redis synchronization." << endl;
	app.runLoop();

	return 0;
}


