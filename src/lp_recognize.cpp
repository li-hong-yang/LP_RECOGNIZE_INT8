#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include "lp_recognize.h"

// test odasf

static Logger gLogger;

using namespace nvinfer1;
using namespace std;

// 加载模型，分配显存和内存
LpRecognize::LpRecognize(const std::string & engine_name)
{
    // select device
    cudaSetDevice(device);

    // load TRT-ENGINE
    std::ifstream file(engine_name, std::ios::binary);
    assert(file.good() == true);
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // build trt context
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);

    // set input output idx
    int inputIndex, outputIndex;
    for (int bi = 0; bi < engine->getNbBindings(); bi++)
    {
        if (engine->bindingIsInput(bi) == true)
        {
            inputIndex = bi;
        }
        else
        {
            outputIndex = bi;
        }
    }
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batch_size * input_c * input_w * input_h *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batch_size * output_size * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    data = new float[batch_size * input_c * input_w * input_h];
    assert(data != nullptr);
    output_buffer = new float[batch_size * output_size];  
    assert(output_buffer != nullptr);  


}


void LpRecognize::preprocess(string& img_name)
{
    cv::Mat img = cv::imread(img_name);
    cv::copyMakeBorder(img,img,2,2,6,6,cv::BORDER_REPLICATE);
    cv::resize(img,img,cv::Size(input_w,input_h));
    int i = 0;
    for (int row = 0; row < input_h; ++row) {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < input_w; ++col) {
            data[0 * 3 * input_h * input_w + i] = ((float)uc_pixel[0]/255.0 - mean) / std ;
            data[0 * 3 * input_h * input_w + i + input_h * input_w] = ((float)uc_pixel[1]/255.0 - mean) / std ;
            data[0 * 3 * input_h * input_w + i + 2 * input_h * input_w] = ((float)uc_pixel[2]/255.0 - mean) / std ;
            uc_pixel += 3;
            ++i;
        }
    }
    cout << "pre_deal_done" << endl;
    

}


void LpRecognize::postprocess()
{
    const int rows = 40;
    const int cols = 84;
    std::vector< std::vector<float> > table_pred(rows, std::vector<float>(cols, 0));
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            table_pred[i][j] = output_buffer[i * cols + j];
        }
    }

    std::vector<int> res(rows);
    for(int i = 0; i < rows; ++i)
    {
        auto max_value = std::max_element(std::begin(table_pred[i]), std::end(table_pred[i]));
        auto max_index = std::distance(std::begin(table_pred[i]), max_value);
        res[i] = max_index;
    }

    std::string results;

    for(int i = 0; i < rows; ++i)
    {
        auto one = res[i];
        if ( one != 0 && ( !( i > 0 &&  one == res[i - 1]) ) )
        {
            results += lp_transchars[one - 1];
        }
    };

    cout << results << endl;



}

void LpRecognize::infer()
{
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], data,  batch_size * input_c * input_w * input_h * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output_buffer, buffers[1], batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cout << "infer_done" << endl;
}

// 释放资源
LpRecognize::~LpRecognize()
{
     
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    
    
}




int main()
{
    string name = "../data/0.jpg"; 
    LpRecognize pred("../models_save/lp_int8.trt");                   
    pred.preprocess(name);  
    pred.infer();
    pred.postprocess();                
    return 0;
}

