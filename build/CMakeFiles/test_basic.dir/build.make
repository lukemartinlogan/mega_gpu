# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_SOURCE_DIR = /home/sislam6/GDS_H/mega_gpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sislam6/GDS_H/mega_gpu/build

# Include any dependencies generated for this target.
include CMakeFiles/test_basic.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_basic.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_basic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_basic.dir/flags.make

CMakeFiles/test_basic.dir/test_basic.cc.o: CMakeFiles/test_basic.dir/flags.make
CMakeFiles/test_basic.dir/test_basic.cc.o: CMakeFiles/test_basic.dir/includes_CUDA.rsp
CMakeFiles/test_basic.dir/test_basic.cc.o: /home/sislam6/GDS_H/mega_gpu/test_basic.cc
CMakeFiles/test_basic.dir/test_basic.cc.o: CMakeFiles/test_basic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sislam6/GDS_H/mega_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test_basic.dir/test_basic.cc.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/test_basic.dir/test_basic.cc.o -MF CMakeFiles/test_basic.dir/test_basic.cc.o.d -x cu -rdc=true -c /home/sislam6/GDS_H/mega_gpu/test_basic.cc -o CMakeFiles/test_basic.dir/test_basic.cc.o

CMakeFiles/test_basic.dir/test_basic.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_basic.dir/test_basic.cc.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test_basic.dir/test_basic.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_basic.dir/test_basic.cc.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_basic
test_basic_OBJECTS = \
"CMakeFiles/test_basic.dir/test_basic.cc.o"

# External object files for target test_basic
test_basic_EXTERNAL_OBJECTS =

CMakeFiles/test_basic.dir/cmake_device_link.o: CMakeFiles/test_basic.dir/test_basic.cc.o
CMakeFiles/test_basic.dir/cmake_device_link.o: CMakeFiles/test_basic.dir/build.make
CMakeFiles/test_basic.dir/cmake_device_link.o: libsimple_lib.a
CMakeFiles/test_basic.dir/cmake_device_link.o: CMakeFiles/test_basic.dir/deviceLinkLibs.rsp
CMakeFiles/test_basic.dir/cmake_device_link.o: CMakeFiles/test_basic.dir/deviceObjects1
CMakeFiles/test_basic.dir/cmake_device_link.o: CMakeFiles/test_basic.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sislam6/GDS_H/mega_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test_basic.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_basic.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_basic.dir/build: CMakeFiles/test_basic.dir/cmake_device_link.o
.PHONY : CMakeFiles/test_basic.dir/build

# Object files for target test_basic
test_basic_OBJECTS = \
"CMakeFiles/test_basic.dir/test_basic.cc.o"

# External object files for target test_basic
test_basic_EXTERNAL_OBJECTS =

test_basic: CMakeFiles/test_basic.dir/test_basic.cc.o
test_basic: CMakeFiles/test_basic.dir/build.make
test_basic: libsimple_lib.a
test_basic: CMakeFiles/test_basic.dir/cmake_device_link.o
test_basic: CMakeFiles/test_basic.dir/linkLibs.rsp
test_basic: CMakeFiles/test_basic.dir/objects1
test_basic: CMakeFiles/test_basic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sislam6/GDS_H/mega_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable test_basic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_basic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_basic.dir/build: test_basic
.PHONY : CMakeFiles/test_basic.dir/build

CMakeFiles/test_basic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_basic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_basic.dir/clean

CMakeFiles/test_basic.dir/depend:
	cd /home/sislam6/GDS_H/mega_gpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sislam6/GDS_H/mega_gpu /home/sislam6/GDS_H/mega_gpu /home/sislam6/GDS_H/mega_gpu/build /home/sislam6/GDS_H/mega_gpu/build /home/sislam6/GDS_H/mega_gpu/build/CMakeFiles/test_basic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_basic.dir/depend

