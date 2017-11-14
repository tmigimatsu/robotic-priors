#include <Python.h>
#include <numpy/arrayobject.h>
#include <string>

#include "CSaiEnv.h"

extern "C" {
	static PyObject *init(PyObject *self, PyObject *args);
	static PyObject *step(PyObject *self, PyObject *args);
	static PyObject *reset(PyObject *self, PyObject *args);
}

static PyObject *init(PyObject *self, PyObject *args) {
	// Get image buffer
	PyArrayObject *obj_buf = PyObject_GetAttr(self, "img_buffer");
	if (obj_buf == nullptr) return nullptr;
	const size_t window_height = obj_buf->dimensions[0];
	const size_t window_width = obj_buf->dimensions[1];

	// Parse arguments
	const char *cstr_world_file, *cstr_robot_file, *cstr_robot_name;
	if (!PyArg_ParseTuple(args, "sss", &cstr_world_file, &cstr_robot_file, &cstr_robot_name)) return nullptr;
	std::string world_file(cstr_world_file);
	std::string robot_file(cstr_robot_file);
	std::string robot_name(cstr_robot_name);

	// Load robot
	cout << "Loading robot: " << robot_file << endl;
	auto robot = make_shared<Model::ModelInterface>(robot_file, Model::rbdl, Model::urdf, false);
	auto sim = std::make_shared<Simulation::SimulationInterface>(world_file, Simulation::sai2simulation, Simulation::urdf, false);
	auto graphics = std::make_shared<Graphics::GraphicsInterface>(world_file, Graphics::chai, Graphics::urdf, true);

	// Start controller app
	cout << "Initializing app with " << robot_name << endl;
	CSaiEnv *sai2_env = new CSaiEnv(move(robot), move(sim), move(graphics), robot_name, window_width, window_height);
	sai2_env->initialize();

	return sai2_env;
}

static PyObject *step(PyObject *self, PyObject *args) {
	// Get sai env
	PyObject *obj_sai2 = PyObject_GetAttr(self, "sai2_env");
	if (obj_sai2 == nullptr) return nullptr;
	CSaiEnv *sai2_env = dynamic_cast<CSaiEnv *>(obj_sai2);

	// Get image buffer
	PyArrayObject *obj_buf = PyObject_GetAttr(self, "img_buffer");
	if (obj_buf == nullptr) return nullptr;
	uint8_t *observation = obj_buf->data;

	// Parse arguments
	PyArrayObject *obj_action;
	if (!PyArg_ParseTuple(args, "O", &obj_action)) return nullptr;
	double *action = (double *)data;

	// Call step function
	double reward;
	char done = sai2_env->step(action, observation, reward, done);

	// Return reward and done; observation is already changed
	return Py_BuildValue("(db)", reward, done);
}

static PyObject *reset(PyObject *self, PyObject *args) {
	// Get sai env
	PyObject *obj_sai2 = PyObject_GetAttr(self, "sai2_env");
	if (obj_sai2 == nullptr) return nullptr;
	CSaiEnv *sai2_env = dynamic_cast<CSaiEnv *>(obj_sai2);

	// Get image buffer
	PyArrayObject *obj_buf = PyObject_GetAttr(self, "img_buffer");
	if (obj_buf == nullptr) return nullptr;
	uint8_t *observation = obj_buf->data;

	// Call reset function
	sai2_env->reset(observation);
	return nullptr;
}
