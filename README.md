# Build this repo

## 1. docker image
    $ cd ${repo_root}  
    $ docker build -t n-point-turn .

## 2. set up  
    $ source setup.sh

## 3. build  
    $ mkdir build
    $ dkb cmake ..
    $ dkb make -j10

Note that `setup.sh` file defines commands to compile some executables with `nvcc`. 
This build method has been deprecated.

## 4. run 
    $ dkb ./main  
    $ dkb ./Test  
    $ dkb ./Test --gtest_filter=FreespaceTests.rotationByShearing