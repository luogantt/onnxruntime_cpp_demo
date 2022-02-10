#set(ONNX_DIR /home/cxj/Desktop/onnx_deploy/onnxruntime_yolo_cpn/onnxruntime-linux-x64-1.4.0)


#set(ONNX_DIR /home/oem/lg/project/onnx_infer/onnx_cpp/onnxruntime_yolo_cpn-master/onnxruntime-linux-x64-1.4.0)
set(ONNX_DIR ../onnxruntime-linux-x64-1.4.0)
# onnxruntime
include_directories(${ONNX_DIR}/include)
link_directories(${ONNX_DIR}/lib)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${INC_DIR} ${OpenCV_INCLUDE_DIRS})
link_directories(${LIB_DIR} ${OpenCV_LIBS_DIR})
link_libraries(${LINK_LIBS} ${OpenCV_LIBS})


add_executable(onnx_yolo_cpn main.cpp)

# 因为有link_directories(${ONNX_DIR}/lib), 所以动态库的路径就知道了.
# target_link_libraries是链接动态库, 一般默认动态库的命名形式是: lib+动态库名.so,
# 比如这里的: libonnxruntime.so, 在CmakeList.txt中写链接动态库的时候, 可以直接
# 只写动态库名, 将lib和后缀名去掉, 比如这里的onnxruntime.
target_link_libraries(onnx_yolo_cpn onnxruntime)

