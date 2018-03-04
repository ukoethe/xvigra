# Ullrich Koethe. Copyright (C) 2018. MIT license

include_directories(${XVIGRA_INCLUDE_DIR} ${DOCTEST_INCLUDE_DIR})

find_package(gtest REQUIRED)

set(XVIGRA_TESTS
    main.cpp
    test_concepts.cpp
    test_distance_transform.cpp
    test_error.cpp
    test_global.cpp
    test_math.cpp
    test_morphology.cpp
    test_slicer.cpp
)

add_executable(test_xvigra ${XVIGRA_TESTS})
target_link_libraries(test_xvigra xtensor gtest)

add_custom_target(xtest COMMAND test_xvigra DEPENDS test_xvigra)
