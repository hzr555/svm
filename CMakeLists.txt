# project name
PROJECT(opencv_read_images)

cmake_minimum_required(VERSION 3.5)

# !!!!
find_package(OpenCV REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_read_images main.cpp)


# .so
link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(opencv_read_images ${OpenCV_LIBRARIES})
