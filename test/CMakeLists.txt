# Ullrich Koethe. Copyright (C) 2018. MIT license

# add_definitions(-DHAVE_CBLAS=1)

# find_package(LAPACK)
# find_package(BLAS)
# message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})  # optional
# message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})  # optional

set(XVIGRA_TESTS
    main.cpp
    test_concepts.cpp
    test_distance_transform.cpp
    test_error.cpp
    test_gaussian.cpp
    test_global.cpp
    test_math.cpp
    test_morphology.cpp
    test_padding.cpp
    test_separable_convolution.cpp
    test_slicer.cpp
    test_splines.cpp
    test_tiny_vector.cpp
)

add_executable(test_xvigra ${XVIGRA_TESTS})

target_link_libraries(test_xvigra xvigra xtensor_io ${BUILD_TESTS})
# target_link_libraries(test_xvigra xvigra xtensor_io ${BUILD_TESTS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_custom_target(xtest COMMAND test_xvigra DEPENDS test_xvigra)
