


CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
NVCC = $(CUDA_BIN_PATH)/nvcc

FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread

NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

INCLUDE = -I/usr/local/include/opencv4 -I$(CUDA_INC_PATH) -I lib/
#INCLUDE = -I/usr/local/include/opencv4 -I$(CUDA_INC_PATH)

LIBS = -L/usr/local/lib -L$(CUDA_LIB_PATH) -lopencv_dnn -lopencv_gapi -lopencv_highgui -lopencv_ml \
 -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_calib3d \
-lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc \
-lopencv_core -lcudart

CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

NVCC_FLAGS = -m64 -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr



sgbm: sgbm.cpp lib/sgbm_helper.cpp cuda.o sgbm_helper.cu.o
	g++ -g -o sgbm -std=c++11 $(INCLUDE) $^ $(LIBS)


%.cu.o: lib/%.cu
	$(NVCC)  $(NVCC_GENCODES) -g -G -c -o $@ $(NVCC_INCLUDE) $<

cuda.o: sgbm_helper.cu.o
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^
clean:
	rm -f *.o lib/*.so