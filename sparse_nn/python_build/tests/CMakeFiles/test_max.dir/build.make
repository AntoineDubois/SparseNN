# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build"

# Include any dependencies generated for this target.
include tests/CMakeFiles/test_max.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test_max.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_max.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_max.dir/flags.make

tests/CMakeFiles/test_max.dir/test_max.cxx.o: tests/CMakeFiles/test_max.dir/flags.make
tests/CMakeFiles/test_max.dir/test_max.cxx.o: /Users/duboisantoine/Library/Mobile\ Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/tests/test_max.cxx
tests/CMakeFiles/test_max.dir/test_max.cxx.o: tests/CMakeFiles/test_max.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_max.dir/test_max.cxx.o"
	cd "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/tests" && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test_max.dir/test_max.cxx.o -MF CMakeFiles/test_max.dir/test_max.cxx.o.d -o CMakeFiles/test_max.dir/test_max.cxx.o -c "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/tests/test_max.cxx"

tests/CMakeFiles/test_max.dir/test_max.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_max.dir/test_max.cxx.i"
	cd "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/tests" && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/tests/test_max.cxx" > CMakeFiles/test_max.dir/test_max.cxx.i

tests/CMakeFiles/test_max.dir/test_max.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_max.dir/test_max.cxx.s"
	cd "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/tests" && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/tests/test_max.cxx" -o CMakeFiles/test_max.dir/test_max.cxx.s

# Object files for target test_max
test_max_OBJECTS = \
"CMakeFiles/test_max.dir/test_max.cxx.o"

# External object files for target test_max
test_max_EXTERNAL_OBJECTS =

tests/test_max: tests/CMakeFiles/test_max.dir/test_max.cxx.o
tests/test_max: tests/CMakeFiles/test_max.dir/build.make
tests/test_max: tests/CMakeFiles/test_max.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_max"
	cd "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/tests" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_max.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_max.dir/build: tests/test_max
.PHONY : tests/CMakeFiles/test_max.dir/build

tests/CMakeFiles/test_max.dir/clean:
	cd "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/tests" && $(CMAKE_COMMAND) -P CMakeFiles/test_max.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_max.dir/clean

tests/CMakeFiles/test_max.dir/depend:
	cd "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn" "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/tests" "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build" "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/tests" "/Users/duboisantoine/Library/Mobile Documents/com~apple~CloudDocs/startup/tests/test_sparse_network/sparse_nn/python_build/tests/CMakeFiles/test_max.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test_max.dir/depend

