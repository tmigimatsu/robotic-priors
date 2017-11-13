#ifndef DEMO_PROJECT_H
#define DEMO_PROJECT_H

// CS225a
#include <redis/RedisClient.h>
#include <timer/LoopTimer.h>

// Standard
#include <string>
#include <thread>

// External
#include <Eigen/Core>
#include <model/ModelInterface.h>
#include <simulation/SimulationInterface.h>
#include <graphics/GraphicsInterface.h>
#include <graphics/ChaiGraphics.h>
#include <GLFW/glfw3.h> //must be loaded after loading opengl/glew


class SaiGym {

public:

	SaiGym(std::shared_ptr<Model::ModelInterface> robot,
	       const std::string &robot_name,
		   std::shared_ptr<Simulation::SimulationInterface> sim,
		   std::shared_ptr<Graphics::GraphicsInterface> graphics) :
		robot(robot),
		dof(robot->dof()),
		kRobotName(robot_name),
		sim(sim),
		graphics(dynamic_cast<Graphics::ChaiGraphics *>(graphics->_graphics_internal)),
		KEY_COMMAND_TORQUES (RedisServer::KEY_PREFIX + robot_name + "::actuators::fgc"),
		KEY_EE_POS          (RedisServer::KEY_PREFIX + robot_name + "::tasks::ee_pos"),
		KEY_EE_POS_DES      (RedisServer::KEY_PREFIX + robot_name + "::tasks::ee_pos_des"),
		KEY_JOINT_POSITIONS (RedisServer::KEY_PREFIX + robot_name + "::sensors::q"),
		KEY_JOINT_VELOCITIES(RedisServer::KEY_PREFIX + robot_name + "::sensors::dq"),
		KEY_TIMESTAMP       (RedisServer::KEY_PREFIX + robot_name + "::timestamp"),
		KEY_KP_POSITION     (RedisServer::KEY_PREFIX + robot_name + "::tasks::kp_pos"),
		KEY_KV_POSITION     (RedisServer::KEY_PREFIX + robot_name + "::tasks::kv_pos"),
		KEY_KP_ORIENTATION  (RedisServer::KEY_PREFIX + robot_name + "::tasks::kp_ori"),
		KEY_KV_ORIENTATION  (RedisServer::KEY_PREFIX + robot_name + "::tasks::kv_ori"),
		KEY_KP_JOINT        (RedisServer::KEY_PREFIX + robot_name + "::tasks::kp_joint"),
		KEY_KV_JOINT        (RedisServer::KEY_PREFIX + robot_name + "::tasks::kv_joint"),
		command_torques_(dof),
		Jv_(3, dof),
		N_(dof, dof),
		g_(dof),
		q_des_(dof),
		dq_des_(dof)
	{
		command_torques_.setZero();

		// Home configuration for Kuka iiwa
		q_des_ << 90, -30, 0, 60, 0, -90, -60;
		q_des_ *= M_PI / 180.0;
		dq_des_.setZero();

		sim->setJointPositions(kRobotName, q_des_);
		sim->setJointVelocities(kRobotName, dq_des_);

		// Desired end effector position
		x_des_ << -0.1, 0.4, 0.7;
		dx_des_.setZero();
	}

	/***** Public functions *****/

	void initialize();
	void runLoop();

protected:

	/***** Enums *****/

	// State enum for controller state machine inside runloop()
	enum ControllerState {
		REDIS_SYNCHRONIZATION,
		JOINT_SPACE_INITIALIZATION,
		OP_SPACE_POSITION_CONTROL
	};

	// Return values from computeControlTorques() methods
	enum ControllerStatus {
		RUNNING,  // Not yet converged to goal position
		FINISHED  // Converged to goal position
	};

	/***** Constants *****/

	const int dof;  // Initialized with robot model
	const double kToleranceInitQ  = 0.1;  // Joint space initialization tolerance
	const double kToleranceInitDq = 0.1;  // Joint space initialization tolerance
	const double kMaxVelocity = 0.5;  // Maximum end effector velocity

	const int kControlFreq = 1000;         // 1 kHz control loop
	const int kSimulationFreq = 10000;         // 1 kHz control loop
	const int kEnvironmentFreq = 10;         // 1 kHz control loop
	const int kInitializationPause = 1e6;  // 1ms pause before starting control loop

	const std::string kRedisHostname = "127.0.0.1";
	const int kRedisPort = 6379;

	const std::string kRobotName;
	const std::string kCameraName = "camera_fixed";
	const int kWindowWidth = 600;
	const int kWindowHeight = 400;

	// Redis keys:
	// - write:
	const std::string KEY_COMMAND_TORQUES;
	const std::string KEY_EE_POS;
	const std::string KEY_EE_POS_DES;
	// - read:
	const std::string KEY_JOINT_POSITIONS;
	const std::string KEY_JOINT_VELOCITIES;
	const std::string KEY_TIMESTAMP;
	const std::string KEY_KP_POSITION;
	const std::string KEY_KV_POSITION;
	const std::string KEY_KP_ORIENTATION;
	const std::string KEY_KV_ORIENTATION;
	const std::string KEY_KP_JOINT;
	const std::string KEY_KV_JOINT;

	/***** Member functions *****/

	void readRedisValues();
	void updateModel();
	void writeRedisValues();
	void publishEnvironment();
	ControllerStatus computeJointSpaceControlTorques();
	ControllerStatus computeOperationalSpaceControlTorques();

	/***** Member variables *****/

	// Robot
	const std::shared_ptr<Model::ModelInterface> robot;
	const std::shared_ptr<Simulation::SimulationInterface> sim;
	Graphics::ChaiGraphics *graphics;

	// Redis
	RedisClient redis_;

	// Timer
	LoopTimer timer_;
	double t_curr_;
	uint64_t controller_counter_ = 0;

	// State machine
	ControllerState controller_state_ = REDIS_SYNCHRONIZATION;

	// Controller variables
	Eigen::VectorXd command_torques_;
	Eigen::MatrixXd Jv_;
	Eigen::MatrixXd N_;
	Eigen::MatrixXd Lambda_x_ = Eigen::MatrixXd(3, 3);
	Eigen::VectorXd g_;
	Eigen::Vector3d x_, dx_;
	Eigen::VectorXd q_des_, dq_des_;
	Eigen::Vector3d x_des_, dx_des_;

	// Graphics
	Eigen::Vector3d camera_pos_;
	Eigen::Vector3d camera_vertical_;
	Eigen::Vector3d camera_lookat_;
	GLFWwindow *window_;
	GLubyte *gl_buffer_ = new GLubyte[3 * kWindowWidth * kWindowHeight];
	Eigen::MatrixXi buffer_r_ = Eigen::MatrixXi(kWindowHeight, kWindowWidth);
	Eigen::MatrixXi buffer_g_ = Eigen::MatrixXi(kWindowHeight, kWindowWidth);
	Eigen::MatrixXi buffer_b_ = Eigen::MatrixXi(kWindowHeight, kWindowWidth);

	// Default gains (used only when keys are nonexistent in Redis)
	double kp_pos_ = 40;
	double kv_pos_ = 10;
	double kp_ori_ = 40;
	double kv_ori_ = 10;
	double kp_joint_ = 40;
	double kv_joint_ = 10;
};

#endif  // DEMO_PROJECT_H
