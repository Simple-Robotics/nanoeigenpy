function(nanoeigenpy_add_test name)
  set(filename ${name}.cpp)
  nanobind_add_module(${name} NB_SHARED NB_SUPPRESS_WARNINGS ${filename})
  target_link_libraries(${name} PRIVATE nanoeigenpy_headers)
endfunction()

nanoeigenpy_add_test(quaternion)
