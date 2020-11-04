PROJECT_SOURCE_DIR=$(cd $(dirname ${BASH_SOURCE[0]:-${(%):-%x}})/; pwd)

# default build tool
build_tool="make"

docker_image="n-point-turn"

#  "docker build"
# --privileged needed because of access NVIDIA GPU 
# Performance Counters on the target device_* error
dkb() {
    echo "running dkb ..."

    CMD="docker run -it --rm\
        --gpus all \
        --privileged \
        -v ${HOME}/.Xauthority:/home/user/.Xauthority \
        -v $PROJECT_SOURCE_DIR:$PROJECT_SOURCE_DIR \
        -w `realpath $PWD` \
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
    cib clang-format -i src/* tests/*
}

build_main() {
    cib nvcc src/main.cu  src/Paper.cu -o main
}

build_tests() {
    cib nvcc tests/tests.cu src/Paper.cu -o test /usr/lib/libgtest.a /usr/lib/libgtest_main.a -lpthread
}

build() {
    format
    dkb nvcc src/main.cu  src/Paper.cu -o main
    dkb nvcc tests/tests.cu src/Paper.cu -o test /usr/lib/libgtest.a /usr/lib/libgtest_main.a -lpthread
}

clean() {
    rm main && rm test
}