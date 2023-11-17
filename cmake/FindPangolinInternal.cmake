
IF(_WIN_)
	FIND_PATH(Glew_INCLUDE_DIRS GL HINTS ${SDK_DIR}/Pangolin/external/glew/include)
	FIND_PATH(Glew_LIB_DIR glew.lib HINTS ${SDK_DIR}/Pangolin/external/glew/lib)
	SET(Glew_DEBUG_LIB ${Glew_LIB_DIR}/glewd.lib)
	SET(Glew_RELEASE_LIB ${Glew_LIB_DIR}/glew.lib)
ELSEIF(_OSX_)
	FIND_PATH(Glew_INCLUDE_DIRS GL HINTS "/usr/local/Cellar/glew/2.2.0_1/include")
	FIND_PATH(Glew_LIB_DIR libGLEW.dylib HINTS "/usr/local/Cellar/glew/2.2.0_1/lib")
	SET(Glew_DEBUG_LIB ${Glew_LIB_DIR}/libGLEW.dylib)
	SET(Glew_RELEASE_LIB ${Glew_LIB_DIR}/libGLEW.dylib)
ENDIF()

ADD_LIBRARY(_glew STATIC IMPORTED)
SET_TARGET_PROPERTIES(_glew PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${Glew_INCLUDE_DIRS}"
		IMPORTED_LOCATION_RELEASE "${Glew_RELEASE_LIB}"
		IMPORTED_LOCATION_DEBUG "${Glew_DEBUG_LIB}"
)

IF(_WIN_)
	FIND_PATH(JPEG_INCLUDE_DIRS jpeglib.h HINTS ${SDK_DIR}/Pangolin/external/libjpeg/include)
	FIND_PATH(JPEG_LIB_DIR jpeg.lib HINTS ${SDK_DIR}/Pangolin/external/libjpeg/lib)
	SET(JPEG_LIBRARY ${JPEG_LIB_DIR}/jpeg.lib)
ELSEIF(_OSX_)
	FIND_PATH(JPEG_INCLUDE_DIRS jpeglib.h HINTS "/usr/local/Cellar/jpeg/9e/include")
	FIND_PATH(JPEG_LIB_DIR libjpeg.dylib HINTS "/usr/local/Cellar/jpeg/9e/lib")
	SET(JPEG_LIBRARY ${JPEG_LIB_DIR}/libjpeg.dylib)
ENDIF()

ADD_LIBRARY(_libjpeg STATIC IMPORTED)
SET_TARGET_PROPERTIES(_libjpeg PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${JPEG_INCLUDE_DIRS}"
		IMPORTED_LOCATION "${JPEG_LIBRARY}"
)

IF(_WIN_)
	FIND_PATH(PNG_INCLUDE_DIRS png.h HINTS ${SDK_DIR}/Pangolin/external/libpng/include)
	FIND_PATH(PNG_LIB_DIR libpng16_static.lib HINTS ${SDK_DIR}/Pangolin/external/libpng/lib)
	SET(PNG_DEBUG_LIB ${PNG_LIB_DIR}/libpng16_staticd.lib)
	SET(PNG_RELEASE_LIB ${PNG_LIB_DIR}/libpng16_static.lib)
ELSEIF(_OSX_)
	FIND_PATH(PNG_INCLUDE_DIRS png.h HINTS "/usr/local/Cellar/libpng/1.6.40/include")
	FIND_PATH(PNG_LIB_DIR libpng.dylib HINTS "/usr/local/Cellar/libpng/1.6.40/lib")
	SET(PNG_DEBUG_LIB ${PNG_LIB_DIR}/libpng.dylib)
	SET(PNG_RELEASE_LIB ${PNG_LIB_DIR}/libpng.dylib)
ENDIF()

ADD_LIBRARY(_libpng STATIC IMPORTED)
SET_TARGET_PROPERTIES(_libpng PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}"
		IMPORTED_LOCATION_RELEASE "${PNG_RELEASE_LIB}"
		IMPORTED_LOCATION_DEBUG "${PNG_DEBUG_LIB}"
)

IF(_WIN_)
	FIND_PATH(ZLIB_INCLUDE_DIRS zlib.h HINTS ${SDK_DIR}/Pangolin/external/zlib/include)
	FIND_PATH(ZLIB_LIB_DIR zlib.lib HINTS ${SDK_DIR}/Pangolin/external/zlib/lib)
	SET(ZLIB_DEBUG_LIB ${ZLIB_LIB_DIR}/zlibstaticd.lib)
	SET(ZLIB_RELEASE_LIB ${ZLIB_LIB_DIR}/zlibstatic.lib)
ELSEIF(_OSX_)
	FIND_PATH(ZLIB_INCLUDE_DIRS zlib.h HINTS "/usr/local/Cellar/zlib/1.3/include")
	FIND_PATH(ZLIB_LIB_DIR libz.dylib HINTS "/usr/local/Cellar/zlib/1.3/lib")
	SET(ZLIB_DEBUG_LIB ${ZLIB_LIB_DIR}/libz.dylib)
	SET(ZLIB_RELEASE_LIB ${ZLIB_LIB_DIR}/libz.dylib)
ENDIF()

ADD_LIBRARY(_zlib STATIC IMPORTED)
SET_TARGET_PROPERTIES(_zlib PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIRS}"
		IMPORTED_LOCATION_RELEASE "${ZLIB_RELEASE_LIB}"
		IMPORTED_LOCATION_DEBUG "${ZLIB_DEBUG_LIB}"
)

IF(_WIN_)
	FIND_PATH(Pangolin_INCLUDE_DIRS pangolin HINTS ${SDK_DIR}/Pangolin/include)
	FIND_PATH(Pangolin_LIB_DIR pangolin.lib HINTS ${SDK_DIR}/Pangolin/lib)
	SET(Pangolin_DEBUG_LIB ${Pangolin_LIB_DIR}/pangolind.lib)
	SET(Pangolin_RELEASE_LIB ${Pangolin_LIB_DIR}/pangolin.lib)
ELSEIF(_OSX_)
	FIND_PATH(Pangolin_DIR PangolinConfig.cmake HINTS "/usr/local/lib/cmake/Pangolin")
	FIND_PACKAGE(Pangolin 0.6 REQUIRED)
ENDIF()

IF(_WIN_)
	ADD_LIBRARY(pangolin STATIC IMPORTED)
	SET_TARGET_PROPERTIES(pangolin PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES "C;CXX"
			INTERFACE_INCLUDE_DIRECTORIES "${Pangolin_INCLUDE_DIRS}"
			IMPORTED_LINK_INTERFACE_LIBRARIES "opengl32;glu32;_glew;_libjpeg;_libpng;_zlib"
			IMPORTED_LOCATION_RELEASE "${Pangolin_RELEASE_LIB}"
			IMPORTED_LOCATION_DEBUG "${Pangolin_DEBUG_LIB}"
	)
ENDIF()