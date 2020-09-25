FROM nvidia/cudagl:10.0-devel 

RUN apt -y update && apt -y install \
freeglut3-dev clang-format libgtest-dev cmake

RUN \
    cd /usr/src/googletest/googletest && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cp libgtest* /usr/lib/ && \
    cd .. && \
    rm -rf build && \
    mkdir /usr/local/lib/googletest && \
    ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a && \
    ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
