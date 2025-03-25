# Add a C++ extension module for tests
function(add_tests_cpp_extension name)
  set(filename ${name}.cpp)
  nanobind_add_module(${name} NB_SHARED NB_SUPPRESS_WARNINGS ${filename})
  target_link_libraries(${name} PRIVATE nanoeigenpy_headers)
  add_dependencies(build_tests ${name})

  add_test(
    NAME "${PROJECT_NAME}-import-${name}"
    COMMAND ${Python_EXECUTABLE} -c "import ${name}"
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${name}>
  )
endfunction()

# Add Python test module
function(add_tests_py_module name)
  set(filename tests/${name}.py)
  set(test_target "${PROJECT_NAME}-${name}")
  string(REPLACE "_" "-" test_target ${test_target})
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
  ADD_PYTHON_UNIT_TEST(${test_target} ${filename} "lib")
  unset(PYTHON_EXECUTABLE)
  set_tests_properties(${test_target} PROPERTIES DEPENDS nanoeigenpy)
endfunction()

add_dependencies(build_tests nanoeigenpy)

add_test(
  NAME "${PROJECT_NAME}-import-extension"
  COMMAND ${Python_EXECUTABLE} -c "import ${PROJECT_NAME}"
  WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

add_tests_cpp_extension(quaternion)
add_tests_py_module(test_quaternion)
