#include <Python.h>

#include <string>
#include <iostream>

#include "CSaiEnv.h"

extern "C" {
	void *init(const char *cstr_world_file, const char *cstr_robot_file, const char *cstr_robot_name,
	           size_t window_width, size_t window_height);
	bool step(void *p_sai2_env, const double *action, uint8_t *observation, double *reward, double *info);
	void reset(void *p_sai2_env, uint8_t *observation);
}

void *init(const char *cstr_world_file, const char *cstr_robot_file, const char *cstr_robot_name,
           size_t window_width, size_t window_height) {
	std::string world_file(cstr_world_file);
	std::string robot_file(cstr_robot_file);
	std::string robot_name(cstr_robot_name);

	// Load robot
	std::cout << "Loading robot: " << robot_file << std::endl;
	auto robot = std::make_shared<Model::ModelInterface>(robot_file, Model::rbdl, Model::urdf, false);
	auto sim = std::make_shared<Simulation::SimulationInterface>(world_file, Simulation::sai2simulation, Simulation::urdf, false);
	auto graphics = std::make_shared<Graphics::GraphicsInterface>(world_file, Graphics::chai, Graphics::urdf, true);

	// Start controller app
	std::cout << "Initializing app with " << robot_name << std::endl;
	CSaiEnv *sai2_env = new CSaiEnv(std::move(robot), std::move(sim), std::move(graphics), robot_name, window_width, window_height);
	sai2_env->initialize();

	return sai2_env;
}

bool step(void *p_sai2_env, const double *action, uint8_t *observation, double *reward, double *info) {
	CSaiEnv *sai2_env = reinterpret_cast<CSaiEnv *>(p_sai2_env);
	return sai2_env->step(action, observation, *reward, info);
}

void reset(void *p_sai2_env, uint8_t *observation) {
	CSaiEnv *sai2_env = reinterpret_cast<CSaiEnv *>(p_sai2_env);
	sai2_env->reset(observation);
}
