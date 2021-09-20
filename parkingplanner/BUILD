load("//bazel/build_rules/cc:defs.bzl", "av_cc_object_library", "dw_cc_module_library")
load("//bazel/build_rules/install:defs.bzl", "av_install")

# ---------------------------------------------------------------------------------------------------------
# genbuild generated (Comment auto-generated: 06/06/2020, 09:33:21)
# DO NOT place custom targets/rules below this block (editing genbuild-generated targets with # keep-* comments is fine)
# ---------------------------------------------------------------------------------------------------------

dw_cc_module_library(
    name = "dw_parkingplanner",
    srcs = [
        "ParkingPlanner2.cpp",
        "ParkingPlanner2.cu",
        "ParkingPlanner_interface.cpp",
    ],
    allowed_3rdparty_libraries = "//src/dw:driveworks_allowed_3rdparty_libraries",
    copts = [
        "-DDW_EXPORTS",  # Set in src/dw/CMakeLists.txt
    ] + select({
        "//bazel/conditions:experimental": [
            "-DSDK_MODULE_INDEX=108",  # Set in src/dw/CMakeLists.txt
        ],
        "//conditions:default": [],
    }),
    cuda_opts = ["-use_fast_math"],
    features = [] + select({
        "//bazel/conditions:drive_linux": [
           # "compiler_warnings_extra",  # Set via SDKConfiguration.cmake in src/dw/CMakeLists.txt
        ],
        "//bazel/conditions:drive_qnx": [],
        "//bazel/conditions:host_linux": [
           # "compiler_warnings_extra",  # Set via SDKConfiguration.cmake in src/dw/CMakeLists.txt
        ],
    }),
    non_transitive_deps = [
        "//src/dw/core:dw_core",
        "//src/dw/experimental/behaviorplanner:dw_experimental_behaviorplanner",
        "//src/dw/experimental/pathunstructured:dw_experimental_pathunstructured",
        "//src/dw/math:dw_math",
        "//src/dw/worldmodel:dw_worldmodel",
    ],
    public_cxx_transitive_deps = [
        #"${PUBLIC_CXX_LIBRARIES}",
    ],
    tags = ["experimental"],
    deps = ["//src/dw/experimental/parkingplanner:dw_parkingplanner_headers"],
)

filegroup(
    name = "dw_parkingplanner_public_headers",
    srcs = ["ParkingPlanner.h"],
    tags = ["experimental"],
    visibility = ["//visibility:public"],
)

av_install(
    name = "install_dw_parkingplanner_public_headers",
    components = ["dw-headers-none"],
    dest_hint = "dw_parkingplanner",
    dest_type = "inc",
    tags = ["experimental"],
    deps = [":dw_parkingplanner_public_headers"],
)

av_cc_object_library(
    name = "dw_parkingplanner_headers",
    hdrs = [
        "Coord3d.hpp",
        "ParkingPlanner2.hpp",
        ":dw_parkingplanner_public_headers",
    ],
    include_prefix = "dw/experimental/parkingplanner",
    tags = ["experimental"],
    visibility = ["//visibility:public"],
)
