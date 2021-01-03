# Description:
#   OpenCV libraries for video/image processing on aarch64 Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
# 'apt-get install libopencv-dev
# on Raspberry Pi OS 64bit
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = glob(
        [
            "lib64/libopencv_core.so",
            "lib64/libopencv_videoio.so",
            "lib64/libopencv_highgui.so",
            "lib64/libopencv_imgcodecs.so",
            "lib64/libopencv_imgproc.so",
            # "lib64/libopencv_calib3d.so",
            # "lib64/libopencv_features2d.so",
            "lib64/libopencv_video.so",
        ],
    ),
    hdrs = glob([
        # For OpenCV 3.x
        # "include/opencv2/**/*.h*",
        # For OpenCV 4.x
        "include/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        # For OpenCV 3.x
        # "include/",
        # For OpenCV 4.x
        "include/opencv4/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
