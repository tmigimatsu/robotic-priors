#include <Python.h>
#include <numpy/arrayobject.h>

#include <string>
#include <iostream>

#include "CSaiEnv.h"

extern "C" {
	static PyObject *init(PyObject *self, PyObject *args);
	static PyObject *step(PyObject *self, PyObject *args);
	static PyObject *reset(PyObject *self, PyObject *args);

	static PyMethodDef ModuleMethods[] = {
		{"init", init, METH_VARARGS, "Initialize the environment"},
		{"step", step, METH_VARARGS, "Run one timestep of the environment's dynamics"},
		{"reset", step, METH_VARARGS, "Reset the environment"},
		{nullptr, nullptr, 0, nullptr}
	};

	static struct PyModuleDef sai2module = {
		PyModuleDef_HEAD_INIT,
		"sai2", // Module name
		nullptr, // module documentation, may be NULL
		-1, // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
		ModuleMethods
	};

	PyMODINIT_FUNC PyInit_sai2(void) {
		return PyModule_Create(&sai2module);
	}
}

static void freeCSaiEnv(PyObject *obj_sai2) {
	void *ptr = PyCapsule_GetPointer(obj_sai2, "CSaiEnv");
	if (ptr == nullptr) return;
	CSaiEnv *sai2_env = reinterpret_cast<CSaiEnv *>(ptr);
	delete sai2_env;
}

static PyObject *init(PyObject *self, PyObject *args) {
	// Get image buffer
	PyObject *obj_buf = PyObject_GetAttrString(self, "img_buffer");
	if (obj_buf == nullptr) return nullptr;
	PyArrayObject *arrobj_buf;
	if (!PyArray_OutputConverter(obj_buf, &arrobj_buf)) return nullptr;
	const size_t window_height = arrobj_buf->dimensions[0];
	const size_t window_width = arrobj_buf->dimensions[1];

	// Parse arguments
	const char *cstr_world_file, *cstr_robot_file, *cstr_robot_name;
	if (!PyArg_ParseTuple(args, "sss", &cstr_world_file, &cstr_robot_file, &cstr_robot_name)) return nullptr;
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

	return PyCapsule_New(sai2_env, "CSaiEnv", freeCSaiEnv);
}

static PyObject *step(PyObject *self, PyObject *args) {
	// Get sai env
	PyObject *obj_sai2 = PyObject_GetAttrString(self, "sai2_env");
	if (obj_sai2 == nullptr) return nullptr;
	void *ptr = PyCapsule_GetPointer(obj_sai2, "CSaiEnv");
	if (ptr == nullptr) return nullptr;
	CSaiEnv *sai2_env = reinterpret_cast<CSaiEnv *>(ptr);

	// Get image buffer
	PyObject *obj_buf = PyObject_GetAttrString(self, "img_buffer");
	if (obj_buf == nullptr) return nullptr;
	PyArrayObject *arrobj_buf;
	if (!PyArray_OutputConverter(obj_buf, &arrobj_buf)) return nullptr;
	uint8_t *observation = reinterpret_cast<uint8_t *>(arrobj_buf->data);

	// Parse arguments
	PyArrayObject *obj_action;
	if (!PyArg_ParseTuple(args, "O&", PyArray_OutputConverter, &obj_action)) return nullptr;
	double *action = (double *)obj_action->data;

	// Call step function
	double reward;
	char done = sai2_env->step(action, observation, reward);

	// Return reward and done; observation is already changed
	return Py_BuildValue("(db)", reward, done);
}

static PyObject *reset(PyObject *self, PyObject *args) {
	// Get sai env
	PyObject *obj_sai2 = PyObject_GetAttrString(self, "sai2_env");
	void *ptr = PyCapsule_GetPointer(obj_sai2, "CSaiEnv");
	if (ptr == nullptr) return nullptr;
	CSaiEnv *sai2_env = reinterpret_cast<CSaiEnv *>(ptr);

	// Get image buffer
	PyObject *obj_buf = PyObject_GetAttrString(self, "img_buffer");
	if (obj_buf == nullptr) return nullptr;
	PyArrayObject *arrobj_buf;
	if (!PyArray_OutputConverter(obj_buf, &arrobj_buf)) return nullptr;
	uint8_t *observation = reinterpret_cast<uint8_t *>(arrobj_buf->data);

	// Call reset function
	sai2_env->reset(observation);
	return nullptr;
}
