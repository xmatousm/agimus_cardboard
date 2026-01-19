cmake_minimum_required(VERSION 3.22.1)

#
# Project definition
#
project(agimus_cardboard LANGUAGES CXX)

#
# Handle dependencies by reading the package.xml
#
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

generate_parameter_module(
  cardboard_parameters
  agimus_cardboard/cardboard_parameters.yaml)

generate_parameter_module(
  hole_planner_parameters
  agimus_cardboard/hole_planner_parameters.yaml)

ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION
                             lib/${PROJECT_NAME})

ament_python_install_package(geometry)
ament_python_install_package(mathlib)

#
# Installation
#
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
#install(DIRECTORY config launch rviz templates
#        DESTINATION share/${PROJECT_NAME})

ament_auto_package()
