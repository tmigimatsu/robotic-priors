# Set executable name
set(EXECUTABLE_NAME sai_gym)

# Create executable
add_executable(${EXECUTABLE_NAME}
	${CMAKE_CURRENT_SOURCE_DIR}/gym_sai2/envs/CSaiEnv.cpp
)

# Link libraries
find_library(GLFW_LIBRARY glfw)
target_link_libraries(${EXECUTABLE_NAME}
	${SAI2_LIBRARIES}
	${GLFW_LIBRARY}
)
