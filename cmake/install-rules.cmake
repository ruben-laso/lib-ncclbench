if(PROJECT_IS_TOP_LEVEL)
  set(
      CMAKE_INSTALL_INCLUDEDIR "include/libncclbench-${PROJECT_VERSION}"
      CACHE STRING ""
  )
  set_property(CACHE CMAKE_INSTALL_INCLUDEDIR PROPERTY TYPE PATH)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package libncclbench)

install(
    DIRECTORY
    include/
    "${PROJECT_BINARY_DIR}/export/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT libncclbench_Development
)

install(
    TARGETS libncclbench_libncclbench
    EXPORT libncclbenchTargets
    RUNTIME #
    COMPONENT libncclbench_Runtime
    LIBRARY #
    COMPONENT libncclbench_Runtime
    NAMELINK_COMPONENT libncclbench_Development
    ARCHIVE #
    COMPONENT libncclbench_Development
    INCLUDES #
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    libncclbench_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
    CACHE STRING "CMake package config location relative to the install prefix"
)
set_property(CACHE libncclbench_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
mark_as_advanced(libncclbench_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${libncclbench_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT libncclbench_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${libncclbench_INSTALL_CMAKEDIR}"
    COMPONENT libncclbench_Development
)

install(
    EXPORT libncclbenchTargets
    NAMESPACE libncclbench::
    DESTINATION "${libncclbench_INSTALL_CMAKEDIR}"
    COMPONENT libncclbench_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
