#ifndef _LP_RECOGNIZE_H_
#define _LP_RECOGNIZE_H_

#include <string>
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include <vector>
using namespace std;

class LpRecognize
{
public:
    LpRecognize(const std::string & engine_name);
    ~LpRecognize(); 
    void preprocess(string& img_name);
    void postprocess();
    void infer();

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    void* buffers[2];           // context input and output
    float* data;                // context input
    float* output_buffer;       // context output

    const int batch_size = 1;
    const int input_c = 3;       // 通道数
    const int input_w = 160;     // 特征向量维数w
    const int input_h = 32;     // 特征向量维数h
    const int device = 0;

    const int output_len = 40;
    const int output_cls = 84;
    const int output_size = 40*84;

    const float std = 0.193;
    const float mean = 0.588;


    const std::vector<std::string> lp_transchars = {
                "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
                "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
                "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",  "2",  "3",  "4",
                "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",
                "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",
                "V",  "W",  "X",  "Y",  "Z",  "港", "学", "使", "警", "澳", "挂", "军",
                "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航", "空"};

};

#endif // _LP_RECOGNIZE_H_
