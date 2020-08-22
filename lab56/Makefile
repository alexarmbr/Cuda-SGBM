# CS 179 Labs 5-6 Makefile
# Written by Aadyot Bhatnagar, 2018

# Input Names
CUDA_FILES = utils.cu
CPP_FILES = model.cpp layers.cpp
CPP_MAIN = main.cpp

# Directory names
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# ------------------------------------------------------------------------------

# CUDA path, compiler, and flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCC_FLAGS := -m32
else
	NVCC_FLAGS := -m64
endif

NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
	      --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_CUDA_LIBS = 
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++11 -pthread
INCLUDE = -I$(CUDA_INC_PATH)
CUDA_LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -lcudnn -lcurand


# ------------------------------------------------------------------------------
# Object files
# ------------------------------------------------------------------------------

# CUDA Object Files
CUDA_OBJ = $(OBJDIR)/cuda.o
CUDA_OBJ_FILES = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CUDA_FILES)))

# C++ Object Files
CPP_OBJ = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CPP_FILES)))
CONV_OBJ = $(addprefix $(OBJDIR)/conv-, $(addsuffix .o, $(CPP_MAIN)))
DENSE_OBJ = $(addprefix $(OBJDIR)/dense-, $(addsuffix .o, $(CPP_MAIN)))

# List of all common objects needed to be linked into the final executable
COMMON_OBJ = $(CPP_OBJ) $(CUDA_OBJ) $(CUDA_OBJ_FILES)


# ------------------------------------------------------------------------------
# Make rules
# ------------------------------------------------------------------------------

# Top level rules
all: dense-neuralnet conv-neuralnet

dense-neuralnet: $(DENSE_OBJ) $(COMMON_OBJ)
	$(GPP) $(FLAGS) -o $(BINDIR)/$@ $(INCLUDE) $^ $(CUDA_LIBS) 

conv-neuralnet: $(CONV_OBJ) $(COMMON_OBJ)
	$(GPP) $(FLAGS) -o $(BINDIR)/$@ $(INCLUDE) $^ $(CUDA_LIBS)


# Compile C++ Source Files
$(DENSE_OBJ): $(addprefix $(SRCDIR)/, $(CPP_MAIN))
	$(GPP) $(FLAGS) -D CONV=0 -c -o $@ $(INCLUDE) $< 

$(CONV_OBJ): $(addprefix $(SRCDIR)/, $(CPP_MAIN))
	$(GPP) $(FLAGS) -D CONV=1 -c -o $@ $(INCLUDE) $< 

$(CPP_OBJ): $(OBJDIR)/%.o : $(SRCDIR)/%
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<


# Compile CUDA Source Files
$(CUDA_OBJ_FILES): $(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $<


# Clean everything including temporary Emacs files
clean:
	rm -f $(BINDIR)/* $(OBJDIR)/*.o $(SRCDIR)/*~ *~


.PHONY: clean all
