
SET(USE_Boost_VERSION "1.68" CACHE STRING "Expected Boost version")
SET_PROPERTY(CACHE USE_Boost_VERSION PROPERTY STRINGS 1.68)

IF(USE_Boost_VERSION EQUAL 1.68)
	FIND_PATH(Boost_INCLUDE_DIR NAMES boost-1_68 HINTS ${SDK_DIR}/Boost/include)
	SET(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR}/boost-1_68)
	FIND_PATH(Boost_LIB_DIR NAMES libboost_atomic-vc141-mt-gd-x64-1_68.lib HINTS ${SDK_DIR}/Boost/lib)
ENDIF()

IF(USE_Boost_VERSION EQUAL 1.68)
  SET(Boost_ATOMIC_LIB_DEBUG ${Boost_LIB_DIR}/libboost_atomic-vc141-mt-gd-x64-1_68.lib)
  SET(Boost_ATOMIC_LIB_RELEASE ${Boost_LIB_DIR}/libboost_atomic-vc141-mt-x64-1_68.lib)
  SET(Boost_CHRONO_LIB_DEBUG ${Boost_LIB_DIR}/libboost_chrono-vc141-mt-gd-x64-1_68.lib)
  SET(Boost_CHRONO_LIB_RELEASE ${Boost_LIB_DIR}/libboost_chrono-vc141-mt-x64-1_68.lib)
  SET(Boost_DATE_TIME_LIB_DEBUG ${Boost_LIB_DIR}/libboost_date_time-vc141-mt-gd-x64-1_68.lib)
  SET(Boost_DATE_TIME_LIB_RELEASE ${Boost_LIB_DIR}/libboost_date_time-vc141-mt-x64-1_68.lib)
  SET(Boost_SYSTEM_LIB_DEBUG ${Boost_LIB_DIR}/libboost_system-vc141-mt-gd-x64-1_68.lib)
  SET(Boost_SYSTEM_LIB_RELEASE ${Boost_LIB_DIR}/libboost_system-vc141-mt-x64-1_68.lib)
  SET(Boost_THREAD_LIB_DEBUG ${Boost_LIB_DIR}/libboost_thread-vc141-mt-gd-x64-1_68.lib)
  SET(Boost_THREAD_LIB_RELEASE ${Boost_LIB_DIR}/libboost_thread-vc141-mt-x64-1_68.lib)
  
  ADD_LIBRARY(Boost::atomic STATIC IMPORTED)
  SET_TARGET_PROPERTIES(Boost::atomic PROPERTIES
	IMPORTED_LOCATION_RELEASE "${Boost_ATOMIC_LIB_RELEASE}"
	IMPORTED_LOCATION_DEBUG "${Boost_ATOMIC_LIB_DEBUG}")
	
  ADD_LIBRARY(Boost::chrono STATIC IMPORTED)
  SET_TARGET_PROPERTIES(Boost::chrono PROPERTIES
	IMPORTED_LOCATION_RELEASE "${Boost_CHRONO_LIB_RELEASE}"
	IMPORTED_LOCATION_DEBUG "${Boost_CHRONO_LIB_DEBUG}")
	
  ADD_LIBRARY(Boost::date_time STATIC IMPORTED)
  SET_TARGET_PROPERTIES(Boost::date_time PROPERTIES
	IMPORTED_LOCATION_RELEASE "${Boost_DATE_TIME_LIB_RELEASE}"
	IMPORTED_LOCATION_DEBUG "${Boost_DATE_TIME_LIB_DEBUG}")
	
  ADD_LIBRARY(Boost::system STATIC IMPORTED)
  SET_TARGET_PROPERTIES(Boost::system PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}"
	IMPORTED_LOCATION_RELEASE "${Boost_SYSTEM_LIB_RELEASE}"
	IMPORTED_LOCATION_DEBUG "${Boost_SYSTEM_LIB_DEBUG}")
	
  ADD_LIBRARY(Boost::thread STATIC IMPORTED)
  SET_TARGET_PROPERTIES(Boost::thread PROPERTIES
	IMPORTED_LOCATION_RELEASE "${Boost_THREAD_LIB_RELEASE}"
	IMPORTED_LOCATION_DEBUG "${Boost_THREAD_LIB_DEBUG}")
  
ENDIF()
