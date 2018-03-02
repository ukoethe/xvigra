# xvigra

VIGRA2 based on xtensor.

## Building From Source

First make sure that you have [CMake](http://www.cmake.org/) and a C++ compiler environment installed.

Then open a terminal, go to the source directory and type the following commands:

    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_PREFIX_PATH="/path/to/xtensor;/path/to/gtest"
    $ make

## Running unit tests

After building this project you may run its unit tests by using these commands:

    $ make xtest
## License

MIT License

