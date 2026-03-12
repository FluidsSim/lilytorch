cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(interp)

find_package(Torch REQUIRED)

add_executable(interp interp.cpp)
target_link_libraries(interp "${TORCH_LIBRARIES}")
set_property(TARGET interp PROPERTY CXX_STANDARD 17)

include_directories(/usr/include/python3.8)
