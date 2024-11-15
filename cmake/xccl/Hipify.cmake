# From https://github.com/ROCm/rccl-tests/blob/develop/src/CMakeLists.txt

find_program(hipify-perl_executable hipify-perl)
set(HIPIFY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipify")


function(hipify SRC_FILE)
  set(HIPIFY_DIR "${CMAKE_BINARY_DIR}/hipified")

  if (NOT DEFINED hipify-perl_executable)
    message(FATAL_ERROR "Unable to find hipify-perl executable")
  endif ()

  if (NOT EXISTS ${SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${SRC_FILE}")
  endif ()

  # Get the basename of the file and its current directory
  get_filename_component(FILE_NAME ${SRC_FILE} NAME)
  get_filename_component(FILE_DIR ${SRC_FILE} DIRECTORY)

  # Get the relative path of the file
  file(RELATIVE_PATH RELATIVE_FILE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ${FILE_DIR})

  # Check the file extension -> change to .cpp if it is a .cu file
  set(HIP_FILE "${HIPIFY_DIR}/${RELATIVE_FILE_DIR}/${FILE_NAME}")
  get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)

  if (${HIP_FILE} MATCHES "\.cu$")
    string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${FILE_NAME})
  endif ()

  # Create a custom command to create hipified source code
  add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR} && ${hipify-perl_executable} -quiet-warnings ${SRC_FILE} > ${HIP_FILE}
      MAIN_DEPENDENCY ${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
  )

  message(STATUS "When building, hypifying: ${SRC_FILE} -> ${HIP_FILE}")

  # set the variable LAST_HIP_FILE
  set(LAST_HIP_FILE ${HIP_FILE} PARENT_SCOPE)
endfunction()