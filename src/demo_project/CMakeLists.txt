# Set executable name
set(EXECUTABLE_NAME demo_project)

# Find source files
file(GLOB_RECURSE ALL_SOURCE_FILES
	*.cpp
)

# Create executable
add_executable(${EXECUTABLE_NAME}
	${ALL_SOURCE_FILES}
)

# Link libraries
target_link_libraries(${EXECUTABLE_NAME}
	${SAI2_LIBRARIES}
)
