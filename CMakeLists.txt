# Works with 3.11 and tested through 3.15
cmake_minimum_required(VERSION 3.11...3.16)

# Project name and a few useful settings. Other commands can pick up the results
project(generateBrittleFractureDataset
    VERSION 0.1
    DESCRIPTION "Cloth simulation Project"
    LANGUAGES CXX)
    
    
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()    
    
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(EXTERN_PROJECTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)
add_subdirectory(${EXTERN_PROJECTS_DIR})



if(WIN32)
	list (APPEND EXTRA_LIBS ${VORO_PLUS_PLUS_LIBRARIES})
    list (APPEND EXTRA_INCLUDES ${VORO_PLUS_PLUS_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME} ${EXTERN_PROJECTS_DIR}/eigen )
else()
	list (APPEND EXTRA_LIBS ${VORO_PLUS_PLUS_LIBRARIES})
    list (APPEND EXTRA_INCLUDES ${VORO_PLUS_PLUS_INCLUDE_DIRS}${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME} ${EXTERN_PROJECTS_DIR}/eigen )
endif()






add_executable(
    ${PROJECT_NAME} 
    src/main.cpp
    src/particles.cpp
    src/weights.cpp
    src/advance.cpp
	src/advanceMassive.cpp
    src/utils.cpp
    include/generateBrittleFractureDataset/extractCrack.h
    include/generateBrittleFractureDataset/particles.h
    include/generateBrittleFractureDataset/grid.h
    include/generateBrittleFractureDataset/weights.h
    include/generateBrittleFractureDataset/advance.h
	include/generateBrittleFractureDataset/advanceMassive.h
    include/generateBrittleFractureDataset/materials.h
    include/generateBrittleFractureDataset/utils.h)
    
    

if(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE /bigobj)
endif()
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_include_directories(${PROJECT_NAME} PRIVATE ${EXTRA_INCLUDES})
target_link_libraries (${PROJECT_NAME} ${EXTRA_LIBS}) 
target_compile_definitions(${PROJECT_NAME} PUBLIC -DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
