cmake_minimum_required(VERSION 3.8)
project(pusher_ghost_filter)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED src/body_fixed_ghost_filter.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PCL_LIBRARIES} Eigen3::Eigen)

add_executable(pusher_ghost_filter_node src/pusher_ghost_filter_node.cpp)
target_include_directories(pusher_ghost_filter_node PRIVATE include ${PCL_INCLUDE_DIRS})
ament_target_dependencies(pusher_ghost_filter_node
  rclcpp
  sensor_msgs
  nav_msgs
  std_srvs
  message_filters
  pcl_conversions
)
target_link_libraries(pusher_ghost_filter_node ${PROJECT_NAME} ${PCL_LIBRARIES} Eigen3::Eigen)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(TARGETS pusher_ghost_filter_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(PCL Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(body_fixed_ghost_filter_test
    test/body_fixed_ghost_filter_test.cpp
  )
  target_include_directories(body_fixed_ghost_filter_test PRIVATE include ${PCL_INCLUDE_DIRS})
  target_link_libraries(body_fixed_ghost_filter_test ${PROJECT_NAME} ${PCL_LIBRARIES} Eigen3::Eigen)
endif()

ament_package()
