#### 说明

本代码是典型的Ubuntu环境下的yolo+cpn的cpu部署代码，请用Clion进行代码查看。为了方便理解，同时也给出了对应的python代码，代码运行的最终结果将保存在results文件夹下。

如果你想直接通过cmake编译查看结果，请按照如下操作：

1. cd ~/onnxruntime_yolo_cpn/
2. mkdir build
3. cd build
4. cmake ..
5. make
6. cd build/src/
7. ./onnx_yolo_cpn

### requirements

1. opencv: 3.4.2
2. onnxruntime: 1.4.0
【注】：opencv环境请实现配置好在系统环境里，onnxruntime在本工程中自带。
