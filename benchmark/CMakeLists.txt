# Ullrich Koethe. Copyright (C) 2018. MIT license

include_directories(${XVIGRA_INCLUDE_DIR})

set(XVIGRA_BENCHMARKS
    main.cpp
    benchmark_tiny_vector.cpp
)

add_executable(benchmark_xvigra ${XVIGRA_BENCHMARKS})
target_link_libraries(benchmark_xvigra xtensor benchmark::benchmark)

add_custom_target(xbench COMMAND benchmark_xvigra DEPENDS benchmark_xvigra)
