cmake_minimum_required(VERSION 2.8 FATAL_ERROR) 
project(catnip) 

# Boost 
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

set(AppSDK $ENV{AMDAPPSDKROOT})
if(AppSDK)
  include_directories($ENV{AMDAPPSDKROOT}/include)
  link_directories($ENV{AMDAPPSDKROOT}/lib/x86)       #use this on 32 bit  systems
  #link_directories($ENV{AMDAPPSDKROOT}/lib/x86_64)    #use this on 64 bit   systems
endif(AppSDK)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/gtest-1.6.0)
include_directories(${PROJECT_SOURCE_DIR}/lib/gtest-1.6.0/include)

set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /D _VARIADIC_MAX=10")
set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Zi /Od /D _VARIADIC_MAX=10")

add_subdirectory(calico) 
add_subdirectory(tabby)

