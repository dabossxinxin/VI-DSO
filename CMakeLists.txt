SET(PROJECT_NAME DSO)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
#set(CMAKE_VERBOSE_MAKEFILE ON)

set(BUILD_TYPE Release)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)

# optional libraries
find_package(LibZip QUIET)
include(FindGlog)
include(FindBoost)
include(FindPangolin)
find_package(OpenCV QUIET)

# flags
add_definitions("-DWIN32")
add_definitions("-DENABLE_SSE")
add_definitions("-D__SSE2__")
add_definitions("-DNOMINMAX")
add_definitions("-D_USE_MATH_DEFINES")
add_definitions("-DHAS_ZIPLIB=1")
add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")

# Sources files
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
)

set(dso_pangolin_SOURCE_FILES 
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp
)

set(dso_opencv_SOURCE_FILES 
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp
)

include_directories(${CSPARSE_INCLUDE_DIR}) 
include_directories(${CHOLMOD_INCLUDE_DIR}) 
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
#include_directories(${GLOG_INCLUDE_DIRS})
#include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${Glew_INCLUDE_DIRS})
include_directories(${JPEG_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIRS})
include_directories(${LIBZIP_INCLUDE_DIR_ZIP}) 
include_directories(${LIBZIP_INCLUDE_DIR_ZIPCONF}) 
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/sse2neon)

add_library(dso ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})
target_link_libraries(dso Google::glog)
add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp)

target_link_libraries(dso_dataset dso ${CSPARSE_LIBRARY} ${LIBZIP_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(dso_dataset Boost::atomic Boost::chrono Boost::date_time Boost::system Boost::thread)
target_link_libraries(dso_dataset debug Pangolin::pangolind optimized Pangolin::pangolin)