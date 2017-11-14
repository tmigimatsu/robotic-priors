# Set executable name
set(EXECUTABLE_NAME sai_gym)

# Create executable
add_executable(${EXECUTABLE_NAME}
	"${CMAKE_CURRENT_SOURCE_DIR}/gym_sai2/envs/CSaiEnv.cpp"
)

# Link libraries
find_library(GLFW_LIBRARY glfw)
target_link_libraries(${EXECUTABLE_NAME}
	${SAI2_LIBRARIES}
	${GLFW_LIBRARY}
)

# Static library for python interface
set(LIBRARY_NAME sai2py)

# Create library
add_library(${LIBRARY_NAME} SHARED
	"${CMAKE_CURRENT_SOURCE_DIR}/gym_sai2/envs/sai2module.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/gym_sai2/envs/CSaiEnv.cpp"
)

# Link libraries
find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME}
	${SAI2_LIBRARIES}
	${GLFW_LIBRARY}
	${PYTHON_LIBRARIES}
)

