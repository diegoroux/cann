cmake_minimum_required(VERSION 3.12)
project(CTensor LANGUAGES C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")

set(SOURCES
	lib/initializations.c
	lib/linear.c
	lib/adam.c
	lib/models.c
	lib/random.c
	lib/fcl.c
	lib/activations.c
	lib/tensor.c
	lib/optimize.c
	lib/loss.c
)

add_library(ctensor SHARED ${SOURCES})
set_target_properties(ctensor PROPERTIES POSITION_INDEPENDENT_CODE ON)