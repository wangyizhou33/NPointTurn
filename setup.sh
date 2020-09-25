PROJECT_SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]:-${(%):-%x}})/; pwd)

# default build tool
build_tool="make"

docker_image="n-point-turn"

#  "docker build"
dkb() {
    echo "running dkb ..."

    CMD="docker run -it --rm\
        --gpus all
        -v ${HOME}/.Xauthority:/home/user/.Xauthority \
        -v $PROJECT_SOURCE_DIR:$PROJECT_SOURCE_DIR \
        -w `realpath $PWD` -u $(id -u):$(id -g)\
        -e DISPLAY \
        --net=host \
        $docker_image \
        bash -c \"$*\""

    echo ${CMD}

    eval ${CMD}
}

#  "ci build"
#  avoid input device 
#  the following flags are absolutely minimum
cib() {
    echo "running cib ..."

    CMD="docker run -i --rm\
        -v $PROJECT_SOURCE_DIR:$PROJECT_SOURCE_DIR -v $PROJECT_BINARY_DIR:$PROJECT_BINARY_DIR \
        -w `realpath $PWD` -u $(id -u):$(id -g)\
        $docker_image \
        bash -c \"$*\""

    echo ${CMD}

    eval ${CMD}
}

format() {
    dkb clang-format -i src/*
    dkb clang-format -i tests/*
}

build_main() {
    format
    dkb nvcc src/main.cu  src/Paper.cu -o main
}

build_tests() {
    format
    dkb nvcc tests/tests.cpp -o test /usr/lib/libgtest.a /usr/lib/libgtest_main.a -lpthread
}

clean() {
    rm main && rm test
}