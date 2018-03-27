# Ullrich Koethe. Copyright (C) 2018. MIT license

# add_definitions(-DHAVE_CBLAS=1)

# find_package(LAPACK)
# find_package(BLAS)
# message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})  # optional
# message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})  # optional

SET(XVIGRA_TEST_DATA
    color_image.tif
)

set(XVIGRA_TESTS
    main.cpp
    test_array_nd.cpp
    test_concepts.cpp
    # test_distance_transform.cpp
    # test_error.cpp
    # test_gaussian.cpp
    test_global.cpp
    test_math.cpp
    # test_morphology.cpp
    # test_padding.cpp
    test_slice.cpp
    # test_splines.cpp
    test_tiny_vector.cpp
)

add_executable(test_xvigra ${XVIGRA_TESTS})

target_compile_options(test_xvigra PRIVATE "-Wreturn-local-addr" "-Wall")
target_link_libraries(test_xvigra xvigra  ${TEST_FRAMEWORK})
# target_link_libraries(test_xvigra xvigra xtensor_io ${BUILD_TESTS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_custom_target(xtest COMMAND test_xvigra DEPENDS test_xvigra)
FOREACH(file ${XVIGRA_TEST_DATA})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file}
                   ${CMAKE_CURRENT_BINARY_DIR}/${file}
                   COPYONLY)
ENDFOREACH(file)
