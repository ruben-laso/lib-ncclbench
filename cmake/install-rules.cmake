if(PROJECT_IS_TOP_LEVEL)
  set(
      CMAKE_INSTALL_INCLUDEDIR "include/ncclbench-${PROJECT_VERSION}"
      CACHE STRING ""
  )
  set_property(CACHE CMAKE_INSTALL_INCLUDEDIR PROPERTY TYPE PATH)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package ncclbench)

install(
    DIRECTORY
    include/
    "${PROJECT_BINARY_DIR}/export/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT ncclbench_Development
)

install(
    TARGETS ncclbench_ncclbench
    EXPORT ncclbenchTargets
    RUNTIME #
    COMPONENT ncclbench_Runtime
    LIBRARY #
    COMPONENT ncclbench_Runtime
    NAMELINK_COMPONENT ncclbench_Development
    ARCHIVE #
    COMPONENT ncclbench_Development
    INCLUDES #
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    ncclbench_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
    CACHE STRING "CMake package config location relative to the install prefix"
)
set_property(CACHE ncclbench_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
mark_as_advanced(ncclbench_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${ncclbench_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT ncclbench_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${ncclbench_INSTALL_CMAKEDIR}"
    COMPONENT ncclbench_Development
)

install(
    EXPORT ncclbenchTargets
    NAMESPACE ncclbench::
    DESTINATION "${ncclbench_INSTALL_CMAKEDIR}"
    COMPONENT ncclbench_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
