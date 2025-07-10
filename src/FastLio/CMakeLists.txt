set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.5)
project(lio_sam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  # pcl library
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
#find_package(Eigen3 REQUIRED)
#将以上修改为以下三行
set(EIGEN3_INCLUDE_DIR /usr/local/include/eigen3)
find_package(Eigen3 3.3.7 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
#将以上修改为以上三行
find_package(libnabo REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake/geographiclib/")
find_package (GeographicLib REQUIRED)

set(Sophus_INCLUDE_DIRS "/usr/local/sophus-template/include")
set(Sophus_LIBRARIES)
find_package( Sophus REQUIRED )

add_message_files(
  DIRECTORY msg
  FILES
#  cloud_info.msg
#  context_info.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS 
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime 
  message_generation
)

# include directories
include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
        ${GeographicLib_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
	src/DiSCo-SLAM/third_parties
)

# link directories
link_directories(
	include
	${PCL_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
    ${GTSAM_LIBRARY_DIRS}
)

###########
## Build ##
###########

#for fushion
# Maximum Clique Solver
add_library(fast_max-clique_finder
	src/DiSCo-SLAM/third_parties/fast_max-clique_finder/src/findClique.h
	src/DiSCo-SLAM/third_parties/fast_max-clique_finder/src/graphIO.h
	src/DiSCo-SLAM/third_parties/fast_max-clique_finder/src/findClique.cpp
	src/DiSCo-SLAM/third_parties/fast_max-clique_finder/src/findCliqueHeu.cpp
	src/DiSCo-SLAM/third_parties/fast_max-clique_finder/src/utils.cpp
	src/DiSCo-SLAM/third_parties/fast_max-clique_finder/src/graphIO.cpp
	)
target_compile_options(fast_max-clique_finder PRIVATE -w)

add_library(scan_context
	src/DiSCo-SLAM/third_parties/scanContext/scanContext.h
	src/DiSCo-SLAM/third_parties/scanContext/scanContext.cpp
	)


# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)

# Fastlio Mapping
add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${Sophus_LIBRARIES} fmt gtsam ${GeographicLib_LIBRARIES})
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})

# Map Fushion
add_executable(${PROJECT_NAME}_mapFusion src/DiSCo-SLAM/mapFusion.cpp)
add_dependencies(${PROJECT_NAME}_mapFusion ${catkin_EXPORTED_TARGETS}  ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_mapFusion ${catkin_LIBRARIES}
		${PCL_LIBRARIES} libnabo::nabo
		gtsam fast_max-clique_finder scan_context)

#catkin_install_python(PROGRAMS scripts/listener.py
#	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
