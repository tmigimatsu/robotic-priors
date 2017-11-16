#ifndef C_SAI_ENV_H
#define C_SAI_ENV_H

// CS225a
#include <redis/RedisClient.h>
#include <timer/LoopTimer.h>

// Standard
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

// External
#include <Eigen/Core>
#include <model/ModelInterface.h>
#include <simulation/SimulationInterface.h>
#include <graphics/GraphicsInterface.h>
#include <graphics/ChaiGraphics.h>
#include <GLFW/glfw3.h> //must be loaded after loading opengl/glew


class CSaiEnv {

public:

	CSaiEnv(std::shared_ptr<Model::ModelInterface> robot,
	        std::shared_ptr<Simulation::SimulationInterface> sim,
	        std::shared_ptr<Graphics::GraphicsInterface> graphics,
	        std::string& robot_name,
			size_t window_width = 400,
			size_t window_height = 300) :
		robot_(robot),
		dof(robot->dof()),
		kRobotName(robot_name),
		sim(sim),
		kWindowWidth(window_width),
		kWindowHeight(window_height),
		thread_graphics(&CSaiEnv::graphicsMain, this, graphics),
		// graphics(dynamic_cast<Graphics::ChaiGraphics *>(graphics->_graphics_internal)),
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
		q_des_(dof),
		dq_des_(dof)
	{
		reset();
	}

	/***** Public functions *****/

	void initialize();
	void runLoop();
	bool step(const double *action, uint8_t *observation, double& reward, double *info);
	void reset(uint8_t *observation);

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
	const int kEnvironmentFreq = 100;         // 1 kHz control loop
	const int kInitializationPause = 1e6;  // 1ms pause before starting control loop

	const std::string kRedisHostname = "127.0.0.1";
	const int kRedisPort = 6379;

	const std::string kRobotName;

	const std::string kCameraName = "camera_fixed";
	const Eigen::Vector3d kCameraPos = Eigen::Vector3d(-0.8, -0.1, 1);
	const Eigen::Vector3d kCameraVertical = Eigen::Vector3d(0, 0, 1);
	const Eigen::Vector3d kCameraLookat = Eigen::Vector3d(0, -0.3, 0.6);
	const int kWindowWidth;
	const int kWindowHeight;

	const double kWallTolerance = 0.02;
	const double kCornerDistance = 0.1;

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

	void reset();
	void readRedisValues();
	void updateModel();
	void writeRedisValues();
	ControllerStatus computeJointSpaceControlTorques();
	ControllerStatus computeOperationalSpaceControlTorques();

	void syncGraphics();
	void graphicsMain(std::shared_ptr<Graphics::GraphicsInterface> graphics);

	/***** Member variables *****/

	// Robot
	const std::shared_ptr<Model::ModelInterface> robot_;
	const std::shared_ptr<Simulation::SimulationInterface> sim;
	Graphics::ChaiGraphics *graphics;

	bool update_graphics_ = false;
	std::mutex mutex_graphics_;
	std::mutex mutex_robot_;
	std::condition_variable cv_;
	std::thread thread_simulator;
	std::thread thread_graphics;

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
	Eigen::MatrixXd J_;
	Eigen::MatrixXd N_;
	Eigen::MatrixXd Lambda_ = Eigen::MatrixXd(6, 6);
	Eigen::VectorXd g_;
	Eigen::Vector3d x_, dx_;
	Eigen::Matrix3d R_des_, R_ee_to_base_;
	Eigen::Vector3d w_;
	Eigen::VectorXd q_des_, dq_des_;
	Eigen::Vector3d x_des_, dx_des_;
	Eigen::Vector3d action_;

	// Graphics
	GLubyte *gl_buffer_ = nullptr;

	// Default gains (used only when keys are nonexistent in Redis)
	double kp_action_ = 100;
	double kp_pos_ = 100;
	double kv_pos_ = 20;
	double kp_ori_ = 200;
	double kv_ori_ = 40;
	double kp_joint_ = 40;
	double kv_joint_ = 10;
};

#endif  // C_SAI_ENV_H
