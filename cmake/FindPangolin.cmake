
FIND_PATH(Glew_INCLUDE_DIRS GL HINTS ${SDK_DIR}/Pangolin/external/glew/include)
FIND_PATH(Glew_LIB_DIR glew.lib HINTS ${SDK_DIR}/Pangolin/external/glew/lib)
SET(Glew_DEBUG_LIB ${Glew_LIB_DIR}/glewd.lib)
SET(Glew_RELEASE_LIB ${Glew_LIB_DIR}/glew.lib)

ADD_LIBRARY(_glew STATIC IMPORTED)
SET_TARGET_PROPERTIES(_glew PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${Glew_INCLUDE_DIRS}"
	IMPORTED_LOCATION_RELEASE "${Glew_RELEASE_LIB}"
	IMPORTED_LOCATION_DEBUG "${Glew_DEBUG_LIB}"
)

FIND_PATH(JPEG_INCLUDE_DIRS jpeglib.h HINTS ${SDK_DIR}/Pangolin/external/libjpeg/include)
FIND_PATH(JPEG_LIB_DIR jpeg.lib HINTS ${SDK_DIR}/Pangolin/external/libjpeg/lib)
SET(JPEG_LIBRARY ${JPEG_LIB_DIR}/jpeg.lib)

ADD_LIBRARY(_libjpeg STATIC IMPORTED)
SET_TARGET_PROPERTIES(_libjpeg PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${JPEG_INCLUDE_DIRS}"
	IMPORTED_LOCATION "${JPEG_LIBRARY}"
)

FIND_PATH(PNG_INCLUDE_DIRS png.h HINTS ${SDK_DIR}/Pangolin/external/libpng/include)
FIND_PATH(PNG_LIB_DIR libpng16_static.lib HINTS ${SDK_DIR}/Pangolin/external/libpng/lib)
SET(PNG_DEBUG_LIB ${PNG_LIB_DIR}/libpng16_staticd.lib)
SET(PNG_RELEASE_LIB ${PNG_LIB_DIR}/libpng16_static.lib)

ADD_LIBRARY(_libpng STATIC IMPORTED)
SET_TARGET_PROPERTIES(_libpng PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}"
	IMPORTED_LOCATION_RELEASE "${PNG_RELEASE_LIB}"
	IMPORTED_LOCATION_DEBUG "${PNG_DEBUG_LIB}"
)

FIND_PATH(ZLIB_INCLUDE_DIRS zlib.h HINTS ${SDK_DIR}/Pangolin/external/zlib/include)
FIND_PATH(ZLIB_LIB_DIR zlib.lib HINTS ${SDK_DIR}/Pangolin/external/zlib/lib)
SET(ZLIB_DEBUG_LIB ${ZLIB_LIB_DIR}/zlibstaticd.lib)
SET(ZLIB_RELEASE_LIB ${ZLIB_LIB_DIR}/zlibstatic.lib)

ADD_LIBRARY(_zlib STATIC IMPORTED)
SET_TARGET_PROPERTIES(_zlib PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIRS}"
	IMPORTED_LOCATION_RELEASE "${ZLIB_RELEASE_LIB}"
	IMPORTED_LOCATION_DEBUG "${ZLIB_DEBUG_LIB}"
)

FIND_PATH(Pangolin_INCLUDE_DIRS pangolin HINTS ${SDK_DIR}/Pangolin/include)
FIND_PATH(Pangolin_LIB_DIR pangolin.lib HINTS ${SDK_DIR}/Pangolin/lib)
SET(Pangolin_DEBUG_LIB ${Pangolin_LIB_DIR}/pangolind.lib)
SET(Pangolin_RELEASE_LIB ${Pangolin_LIB_DIR}/pangolin.lib)

ADD_LIBRARY(Pangolin::pangolin STATIC IMPORTED)
SET_TARGET_PROPERTIES(Pangolin::pangolin PROPERTIES
	IMPORTED_LINK_INTERFACE_LANGUAGES "C;CXX"
	INTERFACE_INCLUDE_DIRECTORIES "${Pangolin_INCLUDE_DIRS}"
	IMPORTED_LINK_INTERFACE_LIBRARIES "opengl32;glu32;_glew;_libjpeg;_libpng;_zlib"
	IMPORTED_LOCATION_RELEASE "${Pangolin_RELEASE_LIB}"
	IMPORTED_LOCATION_DEBUG "${Pangolin_DEBUG_LIB}"
)