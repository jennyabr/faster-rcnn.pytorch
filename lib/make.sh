#!/usr/bin/env bash

echo 'Please write the CUDA architecture of your GPU (e.g. 60 for P100, 61 for 1080Ti, 37 for K80,'
echo 'find other GPUs here: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)'
read cuda_arch_num

CUDA_PATH=/usr/local/cuda/

python setup.py build_ext --inplace
rm -rf build

CUDA_ARCH="-gencode arch=$cuda_arch_num,code=sm_$cuda_arch_num"

# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
python build.py

# compile roi_pooling
cd ../../
cd model/roi_poolers/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_align
cd ../../../
cd model/roi_poolers/roi_align/src
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_crop
cd ../../../
cd model/roi_poolers/roi_crop/src
echo "Compiling roi crop kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py
