# Ullrich Koethe. Copyright (C) 2018. MIT license

SET(XVIGRA_TEST_DATA
    color_image.tif
)

set(XVIGRA_TESTS
    main.cpp
    test_distance_transform.cpp
)

add_executable(test_xvigra ${XVIGRA_TESTS})

target_link_libraries(test_xvigra xvigra ${TEST_FRAMEWORK})

add_custom_target(xtest COMMAND test_xvigra DEPENDS test_xvigra)

FOREACH(file ${XVIGRA_TEST_DATA})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file}
                   ${CMAKE_CURRENT_BINARY_DIR}/${file}
                   COPYONLY)
ENDFOREACH(file)
