
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

include(boost-python.cmake)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

boost_python_module(sqpregpy sqpregpy.cpp)
target_link_libraries(sqpregpy reg_utils sqp_reg)
