cmake_minimum_required(VERSION 3.1)

project(
  contrast
  VERSION 1.0
  LANGUAGES CXX)


find_package(MPI)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(v1_dir "versions/MPI/v1/")
set(v2_dir "versions/MPI/v2/")
set(v2_1_dir "versions/MPI+OpenMP/v2.1/")

add_executable(contrast-times contrast-enhancement.cpp histogram-equalization.cpp contrast-mpi.cpp utils.cpp)
add_executable(contrast-mpi-v1 ${v1_dir}contrast-enhancement.cpp ${v1_dir}histogram-equalization.cpp ${v1_dir}contrast-mpi.cpp ${v1_dir}utils.cpp)
add_executable(contrast-mpi ${v2_dir}contrast-enhancement.cpp ${v2_dir}histogram-equalization.cpp ${v2_dir}contrast-mpi.cpp ${v2_dir}utils.cpp)
add_executable(contrast-mpi-openmp ${v2_1_dir}contrast-enhancement.cpp ${v2_1_dir}histogram-equalization.cpp ${v2_1_dir}contrast-mpi-openmp.cpp ${v2_1_dir}utils.cpp)

target_link_libraries(contrast-times ${MPI_CXX_LIBRARIES})
target_link_libraries(contrast-mpi-v1 ${MPI_CXX_LIBRARIES})
target_link_libraries(contrast-mpi ${MPI_CXX_LIBRARIES})
target_link_libraries(contrast-mpi-openmp ${MPI_CXX_LIBRARIES})
