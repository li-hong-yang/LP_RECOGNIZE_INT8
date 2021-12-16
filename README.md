/home/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/bin/trtexec --onnx=CRNN.onnx --batch=1 --saveEngine=CRNN.engine

## export int8 engine
python utils/convert_trt_quant.py --calib= --onnx=

## int8 model infer
mkdir build
cd build
cmake ..
make -j8
./lp_recognize ../data/0.jpg