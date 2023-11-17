
cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME DSO)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(SetPlatformVarsInternal)
include(FindSuiteParseInternal)
include(FindEigen3Internal)
include(FindLibZipInternal)
include(FindOpenCVInternal)
include(FindPangolinInternal)

if(_WIN_)
    add_definitions("-D_WIN_")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
elseif(_OSX_)
    add_definitions("-D_OSX_")
endif()

add_definitions("-D__SSE2__")
add_definitions("-DNOMINMAX")
add_definitions("-DENABLE_SSE")
add_definitions("-DHAS_ZIPLIB=1")
add_definitions("-D_USE_MATH_DEFINES")
add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")

set(dso_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/IMUPreintegrator.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
  ${PROJECT_SOURCE_DIR}/src/util/systemInput.cpp
)

set(pangolin_SOURCE_FILES 
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp
)

set(opencv_SOURCE_FILES 
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp
)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/sse2neon)

add_library(dso SHARED ${dso_SOURCE_FILES} ${opencv_SOURCE_FILES} ${pangolin_SOURCE_FILES})
target_link_libraries(dso pangolin SuiteSparse OpenCV LibZip)

add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main/main_dso.cpp)
target_link_libraries(dso_dataset dso)

add_executable(dso_test ${PROJECT_SOURCE_DIR}/src/main/main_test.cpp)
target_link_libraries(dso_test dso)