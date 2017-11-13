# Set executable name
set(EXECUTABLE_NAME sai_gym)

# Find source files
file(GLOB_RECURSE ALL_SOURCE_FILES
	*.cpp
)

# Create executable
add_executable(${EXECUTABLE_NAME}
	${ALL_SOURCE_FILES}
)

# Link libraries
find_library(GLFW_LIBRARY glfw)
target_link_libraries(${EXECUTABLE_NAME}
	${SAI2_LIBRARIES}
	${GLFW_LIBRARY}
)
