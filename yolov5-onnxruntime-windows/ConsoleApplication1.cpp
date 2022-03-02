#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>
#include <iostream>  
#include <onnxruntime_cxx_api.h>
#include <assert.h>
#include <vector>
#include <fstream>


using namespace cv;     //当定义这一行后，cv::imread可以直接写成imread
using namespace std;
using namespace Ort;
using namespace cv::dnn;

//String labels_txt_file = "F:\\Pycharm\\PyCharm_Study\\Others\\c++_learning\\C++_Master\\Onnx\\classification\\classification_classes_ILSVRC2012.txt";

//String labels_txt_file = "/home/oem/lg/project/onnx_infer/onnx_cpp/onnxruntime_yolo_cpn-master/weights/classification_classes_ILSVRC2012.txt";

String labels_txt_file = "C://Users//DELL//source//repos//ConsoleApplication1//weights//classification_classes_ILSVRC2012.txt";
vector<String> readClassNames();                  // string对象作为vector对象 

// 图像处理  标准化处理
void PreProcess(const Mat& image, Mat& image_blob)
{
	Mat input;
	image.copyTo(input);


	//数据处理 标准化
	std::vector<Mat> channels, channel_p;
	split(input, channels);
	Mat R, G, B;
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

	B = (B / 255. - 0.406) / 0.225;
	G = (G / 255. - 0.456) / 0.224;
	R = (R / 255. - 0.485) / 0.229;

	channel_p.push_back(R);
	channel_p.push_back(G);
	channel_p.push_back(B);

	Mat outt;
	merge(channel_p, outt);
	image_blob = outt;
}


// 读取txt文件
// 函数的类型是vector<String> 这是opencv内置的方法
// 也可以写成vector<string>   这是string类的方法
//std::vector<String> readClassNames()
std::vector<String> readClassNames()
{
	std::vector<String> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}



int main()         // 返回值为整型带参的main函数. 函数体内使用或不使用argc和argv都可
{

	//environment （设置为VERBOSE（ORT_LOGGING_LEVEL_VERBOSE）时，方便控制台输出时看到是使用了cpu还是gpu执行）

	//新建onnx 环境
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");


	//设置onnx的一些参数
	Ort::SessionOptions session_options;
	// 使用1个线程执行op,若想提升速度，增加线程数
	session_options.SetIntraOpNumThreads(12);
	//	CUDA加速开启(由于onnxruntime的版本太高，无cuda_provider_factory.h的头文件，加速可以使用onnxruntime V1.8的版本)
	 //OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	  // ORT_ENABLE_ALL: 启用所有可能的优化
	//网络图优化
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	//load  model and creat session

#ifdef _WIN32
	//const wchar_t* model_path = L"F:\\Pycharm\\PyCharm_Study\\Others\\c++_learning\\C++_Master\\Onnx\\classification\\vgg16.onnx";

	const wchar_t* model_path = L"C://Users//DELL//source//repos//ConsoleApplication1//weights//vgg16.onnx";

#else
	//const char* model_path = "F:\\Pycharm\\PyCharm_Study\\Others\\c++_learning\\C++_Master\\Onnx\\classification\\vgg16.onnx";
	const char* model_path = "../../weights/vgg16.onnx";
#endif

	printf("Using Onnxruntime C++ API\n");

	Ort::Session session(env, model_path, session_options);
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;


	//model info
	// 获得模型又多少个输入和输出，一般是指对应网络层的数目
	// 一般输入只有图像的话input_nodes为1
	size_t num_input_nodes = session.GetInputCount();
	// 如果是多输出网络，就会是对应输出的数目
	size_t num_output_nodes = session.GetOutputCount();
	printf("Number of inputs = %zu\n", num_input_nodes);
	printf("Number of output = %zu\n", num_output_nodes);
	//获取输入name
	const char* input_name = session.GetInputName(0, allocator);
	std::cout << "input_name:" << input_name << std::endl;
	//获取输出name
	const char* output_name = session.GetOutputName(0, allocator);
	std::cout << "output_name: " << output_name << std::endl;
	// 自动获取维度数量
	auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	std::cout << "input_dims:" << input_dims[0] << std::endl;
	std::cout << "output_dims:" << output_dims[0] << std::endl;
	std::vector<const char*> input_names{ input_name };
	std::vector<const char*> output_names = { output_name };
	std::vector<const char*> input_node_names = { "input.1" };
	std::vector<const char*> output_node_names = { "70" };
	
	cout << "input_names============" << input_names[0] << endl;


	//加载图片
	Mat img = imread("C://Users//DELL//source//repos//ConsoleApplication1//weights//dog.jpg");
	//Mat img = imread("F:\\Pycharm\\PyCharm_Study\\Others\\c++_learning\\C++_Master\\Onnx\\classification\\dog.jpg");
	Mat det1, det2;
	resize(img, det1, Size(256, 256), INTER_AREA);
	det1.convertTo(det1, CV_32FC3);
	PreProcess(det1, det2);         //标准化处理
	Mat blob = dnn::blobFromImage(det2, 1., Size(224, 224), Scalar(0, 0, 0), false, true);
	printf("Load success!\n");

	clock_t startTime, endTime;
	//创建输入tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	std::vector<Ort::Value> input_tensors;
	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size()));
	/*cout << int(input_dims.size()) << endl;*/
	startTime = clock();

	//(score model & input tensor, get back output tensor)
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_names.size(), output_node_names.data(), output_node_names.size());
	endTime = clock();
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
	//除了第一个节点外，其他参数与原网络对应不上程序就会无法执行
	//第二个参数代表输入节点的名称集合
	//第四个参数1代表输入层的数目
	//第五个参数代表输出节点的名称集合
	//最后一个参数代表输出节点的数目
	 // 获取输出(Get pointer to output tensor float values)
	float* floatarr = output_tensors[0].GetTensorMutableData<float>();     // 也可以使用output_tensors.front(); 获取list中的第一个元素变量  list.pop_front(); 删除list中的第一个位置的元素
	// 得到最可能分类输出

	cout << "floatarr=" << floatarr << endl;
 	Mat newarr = Mat_<double>(1, 1000); //定义一个1*1000的矩阵
	for (int i = 0; i < newarr.rows; i++)
	{
		for (int j = 0; j < newarr.cols; j++) //矩阵列数循环
		{
			newarr.at<double>(i, j) = floatarr[j];
		}
	}

	//cout << newarr << endl;
	cout << newarr.size() << endl;

	//vector<String> labels = readClassNames();


	vector<int>  result_name_id;
	for (int n = 0; n < newarr.rows; n++) {
		Point classNumber;
		double classProb;
		Mat probMat = newarr(Rect(0, n, 1000, 1)).clone();
		Mat result = probMat.reshape(1, 1);
		minMaxLoc(result, NULL, &classProb, NULL, &classNumber);
		int classidx = classNumber.x;

		result_name_id.push_back(classidx);
        /***
		printf("\n current image classification : %s, possible : %.2f\n", labels.at(classidx).c_str(), classProb);
		std::cout << "The run time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
		// 显示文本
		putText(img, labels.at(classidx), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1, 1);
		imshow("Image Classification", img);
		waitKey(0);
		***/
	}

	// 计算运行时间
	std::cout << "The run time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

	cout << "the name is " << result_name_id[0] << endl;
	printf("Done!\n");
	//system("pause");
	return 0;
}

