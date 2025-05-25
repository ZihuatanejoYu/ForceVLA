cmake_minimum_required(VERSION 3.5)
project(bison_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  meshes
  launch
  textures
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()