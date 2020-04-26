/**
 * Implementations of each layer that could appear in a neural network
 * @author Aadyot Bhatnagar
 * @date April 22, 2018
 */

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <sstream>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

#include "layers.hpp"
#include "utils.cuh"
#include "helper_cuda.h"


/******************************************************************************/
/*                          GENERIC LAYER SUPERCLASS                          */
/******************************************************************************/

/**
 * If there is a previous layer, initialize this layer's input as the previous
 * layer's' output (as well as the tensor descriptor).
 */
Layer::Layer(Layer *prev, cublasHandle_t cublasHandle,
    cudnnHandle_t cudnnHandle)
{
    this->prev = prev;
    this->cublasHandle = cublasHandle;
    this->cudnnHandle = cudnnHandle;

    CUDNN_CALL( cudnnCreateTensorDescriptor(&in_shape) );
    if (prev)
    {
        cudnnDataType_t dtype;
        int n, c, h, w, nStride, cStride, hStride, wStride;
        CUDNN_CALL( cudnnGetTensor4dDescriptor(prev->get_out_shape(), &dtype,
            &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride) );
        CUDNN_CALL( cudnnSetTensor4dDescriptor(in_shape, CUDNN_TENSOR_NCHW,
            dtype, n, c, h, w) );
    }

    CUDNN_CALL( cudnnCreateTensorDescriptor(&out_shape) );
}

/** Free the memory being used for internal batch and error representations. */
Layer::~Layer()
{
    if (out_batch != in_batch)
        CUDA_CALL(cudaFree(out_batch));

    if (grad_out_batch != grad_in_batch)
        CUDA_CALL(cudaFree(grad_out_batch));

    if (weights)
        CUDA_CALL(cudaFree(weights));

    if (biases)
        CUDA_CALL(cudaFree(biases));

    if (grad_weights)
        CUDA_CALL(cudaFree(grad_weights));

    if (grad_biases)
        CUDA_CALL(cudaFree(grad_biases));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_shape));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_shape));
}

/**
 * Returns the size of the workspace needed by the current layer to implement
 * the specified algorithm. 0 by default, but can be overridden.
 */
size_t Layer::get_workspace_size() const
{
    return 0;
}

/** Sets the workspace (and its size) to the specified values. */
void Layer::set_workspace(float *workspace, size_t workspace_size)
{
    this->workspace = workspace;
    this->workspace_size = workspace_size;
}

/** Returns the shape of the layer's input. */
cudnnTensorDescriptor_t Layer::get_in_shape() const
{
    return in_shape;
}

/** Returns the shape of the layer's output. */
cudnnTensorDescriptor_t Layer::get_out_shape() const
{
    return out_shape;
}

/** Returns the previous layer */
Layer *Layer::get_prev() const
{
    return this->prev;
}

/** Returns the output for the next layer to take in the forward pass. */
float *Layer::get_output_fwd() const
{
    return out_batch;
}

/**
 * Returns the input from the next layer to be passed back during the backward
 * pass. This will be used by the next layer's constructor to initialize its
 * grad_in_batch.
 */
float *Layer::get_input_bwd() const
{
    return grad_out_batch;
}

/* These method definitions are included for polymorphism purposes, but calling
 * either of them from non loss layers is illegal. */
float Layer::get_loss()
{
    assert(false && "Non-loss layer has no loss.");
    return 0;
}

float Layer::get_accuracy()
{
    assert(false && "Non-loss layer does not support accuracy estimates.");
    return 0;
}

/**
 * Allocates space for a minibatch of output data in memory, as well as for the
 * weights and biases associated with this layer (if any). Allocations are done
 * based on layer shape parameters (namely in_shape, out_shape, n_weights, and
 * n_biases).
 *
 * @pre Layer shape parameters have already been set (e.g. in constructor)
 */
void Layer::allocate_buffers()
{
    // The buffers to store the input minibatch in_batch and its derivative
    // grad_in_batch are already stored in the previous layer, so just share a
    // pointer.
    if (prev)
    {
        this->in_batch = prev->get_output_fwd();
        this->grad_in_batch = prev->get_input_bwd();
    }

    // Get the shape of the output
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype,
        &n, &c, &h, &w, &n_stride, &c_stride, &h_stride, &w_stride) );

    // out_batch and grad_out_batch have the same shape as the output
    int out_size = n * c * h * w;
    CUDA_CALL( cudaMalloc(&out_batch, out_size * sizeof(float)) );
    CUDA_CALL( cudaMalloc(&grad_out_batch, out_size * sizeof(float)) );

    // Allocate buffers for the weights and biases (if there are any)
    if (n_weights > 0)
    {
        CUDA_CALL( cudaMalloc(&weights, n_weights * sizeof(float)) );
        CUDA_CALL( cudaMalloc(&grad_weights, n_weights * sizeof(float)) );
    }
    if (n_biases > 0)
    {
        CUDA_CALL( cudaMalloc(&biases, n_biases * sizeof(float)) );
        CUDA_CALL( cudaMalloc(&grad_biases, n_biases * sizeof(float)) );
    }
}

/**
 * Initializes all weights from a uniform distribution bounded between
 * -1/sqrt(input_size) and +1/sqrt(input_size). Initializes all biases to 0.
 */
void Layer::init_weights_biases()
{
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    CUDNN_CALL(cudnnGetTensor4dDescriptor(in_shape, &dtype, &n, &c, &h, &w,
        &n_stride, &c_stride, &h_stride, &w_stride));

    curandGenerator_t gen;
    float minus_half = -0.5;
    float range = 2 / sqrt(static_cast<float>(c * h * w));
    std::random_device rd;
    CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, rd()) );

    if (weights)
    {
        CURAND_CALL( curandGenerateUniform(gen, weights, n_weights) );
        CUBLAS_CALL( cublasSaxpy(cublasHandle, n_weights,
            &minus_half, weights, 1, weights, 1) );
        CUBLAS_CALL( cublasSscal(cublasHandle, n_weights, &range, weights, 1) );
    }

    if (biases)
        cudaMemsetType<float>(biases, 0.0f, n_biases);

    CURAND_CALL( curandDestroyGenerator(gen) );
}


/******************************************************************************/
/*                        INPUT LAYER IMPLEMENTATION                          */
/******************************************************************************/

/** 
 * The output of an input layer is just 
 */
Input::Input(int n, int c, int h, int w,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(nullptr, cublasHandle, cudnnHandle)
{
    // TODO (set 5): set output tensor descriptor out_shape to have format
    //               NCHW, be floats, and have dimensions n, c, h, w

    allocate_buffers();
}

Input::~Input() = default;

/** Input layer does no processing on its input. */
void Input::forward_pass() {}

/** Nothing is behind the input layer. */
void Input::backward_pass(float learning_rate) {}


/******************************************************************************/
/*                        DENSE LAYER IMPLEMENTATION                          */
/******************************************************************************/

/**
 * Flattens the input shape, sets the output shape based on the specified
 * output dimension, and allocates and initializes buffers appropriately.
 */
Dense::Dense(Layer *prev, int out_dim,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    // Get the input shape for the layer and flatten it if needed
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(in_shape, &dtype, &n, &c, &h, &w,
        &n_stride, &c_stride, &h_stride, &w_stride) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(in_shape, CUDNN_TENSOR_NCHW,
        dtype, n, c * h * w, 1, 1) );

    // Initialize the output shape to be N out_size-dimensional vectors
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_shape, CUDNN_TENSOR_NCHW, dtype,
        n, out_dim, 1, 1));

    // Initialize local shape parameters appropriately
    this->batch_size = n;
    this->in_size = c * h * w;
    this->out_size = out_dim;

    // The weights matrix is in_size by out_size, and there are out_size biases
    this->n_weights = in_size * out_size;
    this->n_biases = out_size;
    allocate_buffers();
    init_weights_biases();

    // Allocate a vector of all ones (filled with thrust::fill)
    CUDA_CALL( cudaMalloc(&onevec, batch_size * sizeof(float)) );
    cudaMemsetType<float>(onevec, 1.0f, batch_size);
}

Dense::~Dense()
{
    CUDA_CALL( cudaFree(onevec) );
}

/**
 * A dense layer's forward pass takes the output from the previous layer,
 * multiplies it with this layer's matrix of weights, and then adds the
 * bias vector.
 */
void Dense::forward_pass()
{
    float one = 1.0, zero = 0.0;

    // TODO (set 5): out_batch = weights^T * in_batch (without biases)

    // out_batch += bias * 1_vec^T (to distribute bias to all outputs in
    // this minibatch of data)
    CUBLAS_CALL( cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        out_size, batch_size, 1,
        &one,
        biases, out_size,
        onevec, batch_size,
        &one,
        out_batch, out_size) );
}

/**
 * A dense layer's backward pass computes the gradient of the loss function
 * with respect to its weights, biases, and input minibatch of data. It does
 * so given the gradient with respect to its output, computed by the next
 * layer.
 */
void Dense::backward_pass(float learning_rate)
{
    float one = 1.0, zero = 0.0;

    // TODO (set 5): grad_weights = in_batch * (grad_out_batch)^T

    // grad_biases = grad_out_batch * 1_vec
    CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_N,
        out_size, batch_size,
        &one,
        grad_out_batch, out_size,
        onevec, 1,
        &zero,
        grad_biases, 1) );

    // TODO (set 5): grad_in_batch = W * grad_out_batch
    // Note that grad_out_batch is the next layer's grad_in_batch, and
    // grad_in_batch is the previous layer's grad_out_batch

    // Descend along the gradients of weights and biases using cublasSaxpy
    float eta = -learning_rate;

    // TODO (set 5): weights = weights + eta * grad_weights

    // TODO (set 5): biases = biases + eta * grad_biases
}

/******************************************************************************/
/*                     ACTIVATION LAYER IMPLEMENTATION                        */
/******************************************************************************/

/**
 * Initialize output shape to be the same as input shape, and initialize an
 * activation descriptor as appropriate.
 */
Activation::Activation(Layer *prev, cudnnActivationMode_t activationMode,
    double coef, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    // TODO (set 5): get descriptor of input minibatch, in_shape

    // TODO (set 5): set descriptor of output minibatch, out_shape, to have the
    //               same parameters as in_shape and be ordered NCHW

    allocate_buffers();

    // TODO (set 5): create activation descriptor, and set it to have the given
    //               activationMode, propagate NaN's, and have coefficient coef
}

Activation::~Activation()
{
    // TODO (set 5): destroy the activation descriptor
}

/**
 * Applies the activation on {\link Layer::in_batch} and stores the result in
 * {\link Layer::out_batch}.
 */
void Activation::forward_pass()
{
    float one = 1.0, zero = 0.0;

    // TODO (set 5): apply activation, i.e. out_batch = activation(in_batch)
}

/**
 * Uses the chain rule to compute the gradient wrt the input batch based on the
 * values of {\link Layer::grad_out_batch} (computed by the next layer), as well
 * as {\link Layer::out_batch} and {\link Layer::in_batch} (computed during the
 * forward pass). Stores the result in {\link Layer::grad_in_batch}.
 */
void Activation::backward_pass(float learning_rate)
{
    float one = 1.0, zero = 0.0;

    // TODO (set 5): do activation backwards, i.e. compute grad_in_batch
}

/******************************************************************************/
/*                         CONV LAYER IMPLEMENTATION                          */
/******************************************************************************/

/**
 * Initialize filter descriptor, convolution descriptor, and output shape, as
 * well as algorithms to use for forwards and backwards convolutions. n_kernels
 * is the number of output channels desired, the size of each convolutional
 * kernel is (kernel_size x kernel_size), the stride of the convolution is
 * (stride x stride).
 */
Conv2D::Conv2D(Layer *prev, int n_kernels, int kernel_size, int stride,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    // TODO (set 6): Get the input tensor descriptor in_shape into the variables
    //               declared above

    // Compute nubmer of weights and biases
    this->n_weights = n_kernels * c * kernel_size * kernel_size;
    this->n_biases = n_kernels;

    // TODO (set 6): Create & set a filter descriptor for a float array ordered
    //               NCHW, w/ shape n_kernels x c x kernel_size x kernel_size.
    //               This is class field filter_desc.

    // Set tensor descriptor for biases (to broadcast adding biases)
    CUDNN_CALL( cudnnCreateTensorDescriptor(&bias_desc) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, n_kernels, 1, 1) );

    // TODO (set 6): Create and set a convolution descriptor. This will
    //               correspond to a CUDNN_CONVOLUTION on floats with zero
    //               padding (x and y), a stride (in both x and y) equal to
    //               argument stride, and horizontal and vertical dilation
    //               factors of 1. This is class field conv_desc.

    // Set output shape descriptor
    CUDNN_CALL( cudnnGetConvolution2dForwardOutputDim(conv_desc,
        in_shape, filter_desc, &n, &c, &h, &w) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w) );

    // Get convolution algorithms to use
    CUDNN_CALL( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
        in_shape, filter_desc, conv_desc, out_shape,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo) );
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
        in_shape, out_shape, conv_desc, filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_filter_algo));
    CUDNN_CALL( cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
        filter_desc, out_shape, conv_desc, in_shape,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_data_algo) );

    // Allocate all relevant buffers and initialize filters and biases
    allocate_buffers();
    init_weights_biases();
}

Conv2D::~Conv2D()
{
    CUDNN_CALL( cudnnDestroyTensorDescriptor(bias_desc) );

    // TODO (set 6): Destroy filter_desc and conv_desc
}

/**
 * Determins the largest workspace size needed by any of the convolution
 * algorithms being used and returns it.
 */
size_t Conv2D::get_workspace_size() const
{
    size_t acc = 0, tmp = 0;
    CUDNN_CALL( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
        in_shape, filter_desc, conv_desc, out_shape, fwd_algo, &tmp) );
    acc = std::max(acc, tmp);
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
        in_shape, out_shape, conv_desc, filter_desc, bwd_filter_algo, &tmp));
    acc = std::max(acc, tmp);
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
        filter_desc, out_shape, conv_desc, in_shape, bwd_data_algo, &tmp));
    acc = std::max(acc, tmp);
    return acc;
}

/**
 * Forward pass through a convolution layer performs the convolution and
 * adds the biases.
 */
void Conv2D::forward_pass()
{
    float zero = 0, one = 1;

    // TODO (set 6): Perform convolution forward pass (store in out_batch).
    //               Use class fields workspace and workspace_size for the
    //               workspace related arguments in the function call, and
    //               use fwd_algo for the algorithm.

    CUDNN_CALL( cudnnAddTensor(cudnnHandle,
        &one, bias_desc, biases,
        &one, out_shape, out_batch) );
}

/**
 * Backwards pass through a convolution layer computes gradients with respect
 * to filters, biases, and input minibatch of data. It then descends the
 * gradients with respect to filters and biases.
 */
void Conv2D::backward_pass(float learning_rate)
{
    float zero = 0, one = 1;

    // TODO (set 6): Compute the gradient with respect to the filters/weights
    //               and store them in grad_weights.
    //               Use class fields workspace and workspace_size for the
    //               workspace related arguments in the function call, and use
    //               bwd_filter_algo for the algorithm.


    // Compute the gradient with respect to the biases
    CUDNN_CALL( cudnnConvolutionBackwardBias(cudnnHandle,
        &one, out_shape, grad_out_batch,
        &zero, bias_desc, grad_biases) );


    // TODO (set 6): Compute the gradient with respect to the input data
    //               in_batch, and store it in grad_in_batch.
    //               Use class fields workspace and workspace_size for the
    //               workspace related arguments in the function call and use
    //               bwd_data_algo for the algorithm.


    // Descend along the gradients of the weights and biases using cublasSaxpy
    float eta = -learning_rate;
    
    // TODO (set 6): weights = weights + eta * grad_weights

    // TODO (set 6): biases = biases + eta * grad_biases
}


/******************************************************************************/
/*                      POOLING LAYER IMPLEMENTATION                          */
/******************************************************************************/

/**
 * Allocates a pooling descriptor with a (stride x stride) window and a
 * (stride x stride) stride to do mode-type pooling. Also computes output shape
 * of this operation.
 */
Pool2D::Pool2D(Layer* prev, int stride, cudnnPoolingMode_t mode,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    // TODO (set 6): Create and set pooling descriptor to have the given mode,
    //               propagate NaN's, have window size (stride x stride), have
    //               no padding, and have stride (stride x stride)

    // Set output shape
    int n, c, h, w;
    CUDNN_CALL( cudnnGetPooling2dForwardOutputDim(pooling_desc, in_shape,
        &n, &c, &h, &w) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, n, c, h, w) );

    // Allocate output buffer
    allocate_buffers();
}

Pool2D::~Pool2D()
{
    // TODO (set 6): destroy the pooling descriptor
}

/**
 * Pools the input data minibatch in the forward direction.
 */
void Pool2D::forward_pass()
{
    float zero = 0, one = 1;

    // TODO (set 6): do pooling in forward direction, store in out_batch
}

/**
 * Computes the gradients of the pooling operation wrt the input data minibatch.
 */
void Pool2D::backward_pass(float learning_rate)
{
    float zero = 0, one = 1;

    // TODO (set 6): do pooling backwards, store gradient in grad_in_batch
}


/******************************************************************************/
/*                        GENERIC LOSS ABSTRACT CLASS                         */
/******************************************************************************/

/** Inherits from a {\link Layer}. */
Loss::Loss(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle) {}

Loss::~Loss() = default;

/******************************************************************************/
/*                SOFTMAX + CROSS-ENTROPY LOSS IMPLEMENTATION                 */
/******************************************************************************/
SoftmaxCrossEntropy::SoftmaxCrossEntropy(Layer *prev,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Loss(prev, cublasHandle, cudnnHandle)
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    // TODO (set 5): get descriptor of input minibatch, in_shape, into variables
    //               declared above

    // TODO (set 5): set descriptor of output minibatch, out_shape, to have the
    //               same parameters as in_shape and be ordered NCHW

    allocate_buffers();
}

SoftmaxCrossEntropy::~SoftmaxCrossEntropy() = default;

void SoftmaxCrossEntropy::forward_pass()
{
    float one = 1.0, zero = 0.0;

    // TODO (set 5): do softmax forward pass using accurate softmax and
    //               per instance mode. store result in out_batch.
}

/**
 * The gradient of the softmax/cross-entropy loss function wrt its input
 * minibatch is softmax(X) - Y, where X is the input minibatch, and Y is the
 * associated ground truth output minibatch.
 */
void SoftmaxCrossEntropy::backward_pass(float lr)
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride) );
    int size = n * c * h * w;

    // grad_in_batch = softmax(in_batch) - true_Y
    //               = out_batch         - grad_out_batch
    // all have a length equal to "size" variable defined above
    float minus_one = -1.0;

    // TODO (set 5): first, copy grad_in_batch = out_batch

    // TODO (set 5): set grad_in_batch = grad_in_batch - grad_out_batch using
    //               cublasSaxpy

    // normalize the gradient by the batch size (do it once in the beginning, so
    // we don't have to worry about it again later)
    float scale = 1.0f / static_cast<float>(n);
    CUBLAS_CALL( cublasSscal(cublasHandle, size, &scale, grad_in_batch, 1) );
}

/**
 * @return    The cross-entropy loss between our model's probabilistic predictions
 *            (based on the softmax function) and the ground truth.
 *
 * @pre A forward pass has been run (so {\link Layer::out_batch} contains our
 *        predictions on some input batch) and {\link Layer::grad_out_batch}
 *        contains the ground truth we want to predict.
 */
float SoftmaxCrossEntropy::get_loss()
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride) );

    loss = CrossEntropyLoss(out_batch, grad_out_batch, n, c, h, w);
    return loss;
}

/**
 * @return    The accuracy of the model's predictions on a minibatch, given the
 *            ground truth. Here, we treat the model's prediction as the class
 *            assigned the highest probability by softmax.
 *
 * @pre A forward pass has been run (so {\link Layer::out_batch} contains our
 *        predictions on some input batch) and {\link Layer::grad_out_batch}
 *        contains the ground truth we want to predict.
 */
float SoftmaxCrossEntropy::get_accuracy()
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    CUDNN_CALL(cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride));

    acc = SoftThresholdAccuracy(out_batch, grad_out_batch, n, c, h, w);
    return acc;
}
