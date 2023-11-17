
IF(_WIN_)
	FIND_PATH(GLOG_INCLUDE_DIRS NAMES glog HINTS ${CMAKE_SOURCE_DIR}/../../SDK/glog/include)
	FIND_PATH(GLOG_LIB_DIR NAMES glog.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/glog/lib)
	SET(GLOG_RELEASE_LIB ${GLOG_LIB_DIR}/glog.lib)
	SET(GLOG_DEBUG_LIB ${GLOG_LIB_DIR}/glogd.lib)
ELSEIF(_OSX_)
	FIND_PATH(GLOG_INCLUDE_DIRS NAMES glog HINTS "/usr/local/Cellar/glog/0.6.0/include")
	FIND_PATH(GLOG_LIB_DIR NAMES libglog.dylib HINTS "/usr/local/Cellar/glog/0.6.0/lib")
	SET(GLOG_LIB ${GLOG_LIB_DIR}/libglog.dylib)
ENDIF()

IF(_WIN_)
	FIND_PATH(GFLAGS_INCLUDE_DIRS NAMES gflags HINTS ${CMAKE_SOURCE_DIR}/../../SDK/gflags/include)
	FIND_PATH(GFLAGS_LIB_DIR NAMES gflags.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/gflags/lib)
	SET(GFLAGS_RELEASE_LIBS ${GFLAGS_LIB_DIR}/gflags.lib )
	SET(GFLAGS_NOTHREADS_RELEASE_LIBS ${GFLAGS_LIB_DIR}/gflags_nothreads.lib)
	SET(GFLAGS_DEBUG_LIBS ${GFLAGS_LIB_DIR}/gflags_debug.libs)
	SET(GFLAGS_NOTHREADS_DEBUG_LIBS ${GFLAGS_LIB_DIR}/gflags_nothreads_debug.lib)
ELSEIF(_OSX_)
	FIND_PATH(GFLAGS_INCLUDE_DIRS NAMES gflags HINTS "/usr/local/Cellar/gflags/2.2.2/include")
	FIND_PATH(GFLAGS_LIB_DIR NAMES libgflags.dylib HINTS "/usr/local/Cellar/gflags/2.2.2/lib")
	SET(GFLAGS_LIBS ${GFLAGS_LIB_DIR}/libgflags.dylib )
	SET(GFLAGS_NOTHREADS_LIBS ${GFLAGS_LIB_DIR}/libgflags_nothreads.dylib)
ENDIF()

IF(_WIN_)
	ADD_LIBRARY(_gflags SHARED IMPORTED)
	SET_TARGET_PROPERTIES(_gflags PROPERTIES
			IMPORTED_LOCATION_RELEASE "${GFLAGS_RELEASE_LIBS}"
			IMPORTED_LOCATION_DEBUG "${GFLAGS_DEBUG_LIBS}"
	)

	ADD_LIBRARY(_gflags_nothreads SHARED IMPORTED)
	SET_TARGET_PROPERTIES(_gflags_nothreads PROPERTIES
			IMPORTED_LOCATION_RELEASE "${GFLAGS_NOTHREADS_RELEASE_LIBS}"
			IMPORTED_LOCATION_DEBUG "${GFLAGS_NOTHREADS_DEBUG_LIBS}"
	)

	ADD_LIBRARY(Google::glog SHARED IMPORTED)
	SET_TARGET_PROPERTIES(Google::glog PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES "C;CXX"
			INTERFACE_INCLUDE_DIRECTORIES "${GFLAGS_INCLUDE_DIRS};${GLOG_INCLUDE_DIRS}"
			IMPORTED_LINK_INTERFACE_LIBRARIES "_gflags;_gflags_nothreads"
			IMPORTED_LOCATION_RELEASE "${GLOG_RELEASE_LIB}"
			IMPORTED_LOCATION_DEBUG "${GLOG_DEBUG_LIB}"
	)
ELSEIF(_OSX_)
	ADD_LIBRARY(_gflags SHARED IMPORTED)
	SET_TARGET_PROPERTIES(_gflags PROPERTIES
			IMPORTED_LOCATION "${GFLAGS_LIBS}"
	)

	ADD_LIBRARY(_gflags_nothreads SHARED IMPORTED)
	SET_TARGET_PROPERTIES(_gflags_nothreads PROPERTIES
			IMPORTED_LOCATION "${GFLAGS_NOTHREADS_LIBS}"
	)

	ADD_LIBRARY(Google::glog SHARED IMPORTED)
	SET_TARGET_PROPERTIES(Google::glog PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES "C;CXX"
			INTERFACE_INCLUDE_DIRECTORIES "${GFLAGS_INCLUDE_DIRS};${GLOG_INCLUDE_DIRS}"
			IMPORTED_LINK_INTERFACE_LIBRARIES "_gflags"
			IMPORTED_LOCATION "${GLOG_LIB}"
	)

	ADD_LIBRARY(Google::gflags SHARED IMPORTED)
	SET_TARGET_PROPERTIES(Google::gflags PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES "C;CXX"
			INTERFACE_INCLUDE_DIRECTORIES "${GFLAGS_INCLUDE_DIRS}"
			IMPORTED_LOCATION "${GFLAGS_LIBS}"
	)
ENDIF()