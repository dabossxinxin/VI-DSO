
FIND_PATH(OpenCV_INCLUDE_DIR NAMES opencv2 HINTS ${CMAKE_SOURCE_DIR}/../../SDK/opencv/include)
FIND_FILE(OpenCV_DEBUG_LIB opencv_world346d.lib)
FIND_FILE(OpenCV_RELEASE_LIB opencv_world346.lib)

ADD_LIBRARY(OpenCV STATIC IMPORTED)
SET_TARGET_PROPERTIES(OpenCV PROPERTIES
	IMPORTED_LINK_INTERFACE_LANGUAGES "C;CXX"
	INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIR};"
	IMPORTED_LINK_INTERFACE_LIBRARIES "_gflags;_gflags_nothreads"
	IMPORTED_LOCATION_RELEASE "${OpenCV_RELEASE_LIB}"
	IMPORTED_LOCATION_DEBUG "${OpenCV_DEBUG_LIB}"
)