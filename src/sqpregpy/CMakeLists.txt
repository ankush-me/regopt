
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

include(boost-python.cmake)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

boost_python_module(sqpregpy sqpregpy.cpp)
target_link_libraries(sqpregpy reg_utils sqp_reg)

boost_python_module(shape_context pyshape_context.cpp)
target_link_libraries(shape_context reg_utils)
