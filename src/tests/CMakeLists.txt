
add_executable(reg test_sqpreg.cpp)
target_link_libraries(reg sqp_reg sco reg_utils)


include_directories(${PYTHON_INCLUDE_PATH}
    	 			${Boost_INCLUDE_DIRS})
add_executable(test_shape_context test_shape_context)
target_link_libraries(test_shape_context shape_context)
