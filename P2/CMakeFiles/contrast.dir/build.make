# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alumnos/a0405959/CAP2022

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alumnos/a0405959/CAP2022

# Include any dependencies generated for this target.
include CMakeFiles/contrast.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/contrast.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/contrast.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/contrast.dir/flags.make

CMakeFiles/contrast.dir/contrast-enhancement.cpp.o: CMakeFiles/contrast.dir/flags.make
CMakeFiles/contrast.dir/contrast-enhancement.cpp.o: contrast-enhancement.cpp
CMakeFiles/contrast.dir/contrast-enhancement.cpp.o: CMakeFiles/contrast.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alumnos/a0405959/CAP2022/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/contrast.dir/contrast-enhancement.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/contrast.dir/contrast-enhancement.cpp.o -MF CMakeFiles/contrast.dir/contrast-enhancement.cpp.o.d -o CMakeFiles/contrast.dir/contrast-enhancement.cpp.o -c /home/alumnos/a0405959/CAP2022/contrast-enhancement.cpp

CMakeFiles/contrast.dir/contrast-enhancement.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/contrast.dir/contrast-enhancement.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alumnos/a0405959/CAP2022/contrast-enhancement.cpp > CMakeFiles/contrast.dir/contrast-enhancement.cpp.i

CMakeFiles/contrast.dir/contrast-enhancement.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/contrast.dir/contrast-enhancement.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alumnos/a0405959/CAP2022/contrast-enhancement.cpp -o CMakeFiles/contrast.dir/contrast-enhancement.cpp.s

CMakeFiles/contrast.dir/histogram-equalization.cpp.o: CMakeFiles/contrast.dir/flags.make
CMakeFiles/contrast.dir/histogram-equalization.cpp.o: histogram-equalization.cpp
CMakeFiles/contrast.dir/histogram-equalization.cpp.o: CMakeFiles/contrast.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alumnos/a0405959/CAP2022/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/contrast.dir/histogram-equalization.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/contrast.dir/histogram-equalization.cpp.o -MF CMakeFiles/contrast.dir/histogram-equalization.cpp.o.d -o CMakeFiles/contrast.dir/histogram-equalization.cpp.o -c /home/alumnos/a0405959/CAP2022/histogram-equalization.cpp

CMakeFiles/contrast.dir/histogram-equalization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/contrast.dir/histogram-equalization.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alumnos/a0405959/CAP2022/histogram-equalization.cpp > CMakeFiles/contrast.dir/histogram-equalization.cpp.i

CMakeFiles/contrast.dir/histogram-equalization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/contrast.dir/histogram-equalization.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alumnos/a0405959/CAP2022/histogram-equalization.cpp -o CMakeFiles/contrast.dir/histogram-equalization.cpp.s

CMakeFiles/contrast.dir/contrast-mpi.cpp.o: CMakeFiles/contrast.dir/flags.make
CMakeFiles/contrast.dir/contrast-mpi.cpp.o: contrast-mpi.cpp
CMakeFiles/contrast.dir/contrast-mpi.cpp.o: CMakeFiles/contrast.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alumnos/a0405959/CAP2022/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/contrast.dir/contrast-mpi.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/contrast.dir/contrast-mpi.cpp.o -MF CMakeFiles/contrast.dir/contrast-mpi.cpp.o.d -o CMakeFiles/contrast.dir/contrast-mpi.cpp.o -c /home/alumnos/a0405959/CAP2022/contrast-mpi.cpp

CMakeFiles/contrast.dir/contrast-mpi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/contrast.dir/contrast-mpi.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alumnos/a0405959/CAP2022/contrast-mpi.cpp > CMakeFiles/contrast.dir/contrast-mpi.cpp.i

CMakeFiles/contrast.dir/contrast-mpi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/contrast.dir/contrast-mpi.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alumnos/a0405959/CAP2022/contrast-mpi.cpp -o CMakeFiles/contrast.dir/contrast-mpi.cpp.s

CMakeFiles/contrast.dir/utils.cpp.o: CMakeFiles/contrast.dir/flags.make
CMakeFiles/contrast.dir/utils.cpp.o: utils.cpp
CMakeFiles/contrast.dir/utils.cpp.o: CMakeFiles/contrast.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alumnos/a0405959/CAP2022/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/contrast.dir/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/contrast.dir/utils.cpp.o -MF CMakeFiles/contrast.dir/utils.cpp.o.d -o CMakeFiles/contrast.dir/utils.cpp.o -c /home/alumnos/a0405959/CAP2022/utils.cpp

CMakeFiles/contrast.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/contrast.dir/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alumnos/a0405959/CAP2022/utils.cpp > CMakeFiles/contrast.dir/utils.cpp.i

CMakeFiles/contrast.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/contrast.dir/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alumnos/a0405959/CAP2022/utils.cpp -o CMakeFiles/contrast.dir/utils.cpp.s

# Object files for target contrast
contrast_OBJECTS = \
"CMakeFiles/contrast.dir/contrast-enhancement.cpp.o" \
"CMakeFiles/contrast.dir/histogram-equalization.cpp.o" \
"CMakeFiles/contrast.dir/contrast-mpi.cpp.o" \
"CMakeFiles/contrast.dir/utils.cpp.o"

# External object files for target contrast
contrast_EXTERNAL_OBJECTS =

contrast: CMakeFiles/contrast.dir/contrast-enhancement.cpp.o
contrast: CMakeFiles/contrast.dir/histogram-equalization.cpp.o
contrast: CMakeFiles/contrast.dir/contrast-mpi.cpp.o
contrast: CMakeFiles/contrast.dir/utils.cpp.o
contrast: CMakeFiles/contrast.dir/build.make
contrast: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
contrast: /usr/lib/x86_64-linux-gnu/libmpich.so
contrast: CMakeFiles/contrast.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alumnos/a0405959/CAP2022/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable contrast"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/contrast.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/contrast.dir/build: contrast
.PHONY : CMakeFiles/contrast.dir/build

CMakeFiles/contrast.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/contrast.dir/cmake_clean.cmake
.PHONY : CMakeFiles/contrast.dir/clean

CMakeFiles/contrast.dir/depend:
	cd /home/alumnos/a0405959/CAP2022 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alumnos/a0405959/CAP2022 /home/alumnos/a0405959/CAP2022 /home/alumnos/a0405959/CAP2022 /home/alumnos/a0405959/CAP2022 /home/alumnos/a0405959/CAP2022/CMakeFiles/contrast.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/contrast.dir/depend

