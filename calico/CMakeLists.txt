# Project Includes
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/calico/src) 

# Source Project
file(GLOB_RECURSE SRC src/*.h src/*.cpp src/*.cl)

add_library(calico ${SRC})
target_link_libraries(calico OpenCL.lib)

# Testing Project
file(GLOB TEST_SRC test/*.h test/*.cpp)

add_executable(calico_test ${TEST_SRC})
target_link_libraries(calico_test calico gtest gtest_main)

# Experimental Project
file(GLOB EXP_SRC exp/*.h exp/*.cpp)

add_executable(calico_exp ${EXP_SRC})
target_link_libraries(calico_exp calico)
