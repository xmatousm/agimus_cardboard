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

function(generate_parameter_module_with_source_symlink module_name yaml_file)
  generate_parameter_module(${module_name} ${yaml_file})

  get_filename_component(yaml_absolute_path "${yaml_file}" ABSOLUTE
                         BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  get_filename_component(yaml_directory "${yaml_absolute_path}" DIRECTORY)

  set(source_module "${yaml_directory}/${module_name}.py")
  set(generated_module
      "${CMAKE_CURRENT_BINARY_DIR}/generate_parameter_module_temp/${module_name}.py")
  file(RELATIVE_PATH source_module_target "${yaml_directory}"
       "${generated_module}")

  if(EXISTS "${source_module}" AND NOT IS_SYMLINK "${source_module}")
    message(FATAL_ERROR "${source_module} exists and is not a symlink")
  endif()

  add_custom_command(
    OUTPUT "${source_module}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${source_module}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${source_module_target}"
            "${source_module}"
    DEPENDS "${generated_module}"
    VERBATIM)

  add_custom_target(${module_name}_source_module_link ALL
                    DEPENDS "${source_module}")
  add_dependencies(${module_name}_source_module_link ${module_name})
endfunction()

generate_parameter_module_with_source_symlink(
  camera_parameters
  agimus_cardboard/camera_parameters.yaml)

generate_parameter_module_with_source_symlink(
  detector_parameters
  agimus_cardboard/detector_parameters.yaml)

generate_parameter_module_with_source_symlink(
  hole_planner_parameters
  agimus_cardboard/hole_planner_parameters.yaml)

ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION
                             lib/${PROJECT_NAME})

ament_python_install_package(geometry)
ament_python_install_package(mathlib)

ament_auto_package(INSTALL_TO_SHARE templates)
