/**
 * Header file defining each layer that could appear in a neural network
 * @author Aadyot Bhatnagar
 * @date April 22, 2018
 */

#pragma once

#include <cudnn.h>
#include <cublas_v2.h>

/**
 * Generic layer superclass. All other layers are subclasses of this one. Pure
 * virtual functions for the forwards and backwards pass must be implmented by
 * any subclass.
 */
class Layer
{
public:
    Layer(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    virtual ~Layer();
    Layer *get_prev() const;

    /**
     * Computes this layer's forward pass on an input minibatch of data, and
     * passes it to the next layer in the neural network.
     */
    virtual void forward_pass() = 0;

    /**
     * Propagates this layer's gradients with respect to the given data minibatch
     * backwards through the neural network.
     */
    virtual void backward_pass(float learning_rate) = 0;

    /* Functions for allocating and setting up weights, biases, and input/output
     * minibatches of data. */
    void allocate_buffers();
    void init_weights_biases();

    /* Functions to evaluate model performance. Only valid for loss layers,
     * result in an assertion error if called from a non loss layer. */
    virtual float get_loss();
    virtual float get_accuracy();

    /* Valid getters for all layers */
    float *get_output_fwd() const;
    float *get_input_bwd() const;
    cudnnTensorDescriptor_t get_in_shape() const;
    cudnnTensorDescriptor_t get_out_shape() const;

    /* Methods used to set up a cuDNN workspace for faster convolutions */
    virtual size_t get_workspace_size() const;
    void set_workspace(float *workspace, size_t workspace_size);

protected:
    /** Previous layer. */
    Layer *prev;

    /**
     * Tensor descriptor for the input minibatch {\link Layer::in_batch} and
     * its gradient {\link Layer::grad_in_batch}.
     */
    cudnnTensorDescriptor_t in_shape;

    /**
     * Tensor descriptor for the output minibatch {\link Layer::out_batch} and
     * its gradient {\link Layer::grad_out_batch}.
     */
    cudnnTensorDescriptor_t out_shape;

    /** cuBLAS library context */
    cublasHandle_t cublasHandle;

    /** cuDNN library context */
    cudnnHandle_t cudnnHandle;


    /************************************************************************/
    /*                INTERMEDIATE QUANTITIES AND GRADIENTS                 */
    /************************************************************************/

    /** Workspace for cuDNN scratch work during convolutions */
    float *workspace = nullptr;

    /** Size of {\link Layer::workspace} in bytes */
    size_t workspace_size = 0;

    /**
     * The input minibatch to this layer. Computed by the previous layer.
     * Has shape in_shape.
     */
    float *in_batch = nullptr;

    /**
     * The output minibatch from this layer. Pointer shared with in_batch of
     * the next layer.
     * Has shape out_shape.
     */
    float *out_batch = nullptr;


    /**
     * Derivative wrt out_batch. Computed by the next layer.
     * Has shape out_shape.
     */
    float *grad_out_batch = nullptr;

    /**
     * Derivative wrt the input batch.
     * Has shape in_shape.
     */
    float *grad_in_batch = nullptr;


    /************************************************************************/
    /*                MODEL PARAMETERS AND THEIR GRADIENTS                  */
    /************************************************************************/

    /** Model weights */
    float *weights = nullptr;

    /** Model biases */
    float *biases = nullptr;

    /** Derivative wrt the weights */
    float *grad_weights = nullptr;

    /** Derivative wrt the biases */
    float *grad_biases = nullptr;

    /** The number of weights in this layer */
    int n_weights = 0;

    /** The number of biases in this layer */
    int n_biases = 0;
};


/**
 * Implements a layer that takes an input. Must be the first layer of any
 * network.
 */
class Input : public Layer
{
public:
    Input(int n, int c, int h, int w,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Input();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;
};


/**
 * Implements a densely connected layer.
 */
class Dense : public Layer
{
public:
    Dense(Layer *prev, int out_dim,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Dense();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    /** The size of the input vector. */
    int in_size;

    /** The size of an output vector. */
    int out_size;

    /** The size of the data minibatch to be processed per pass. */
    int batch_size;

    /** A vector of all ones. */
    float *onevec;
};


/**
 * Implements an activation supported by cuDNN.
 */
class Activation : public Layer
{
public:
    Activation(Layer *prev, cudnnActivationMode_t activationMode, double coef,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Activation();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    /** Descriptor of the activation implemented by this layer */
    cudnnActivationDescriptor_t activation_desc;
};


/**
 * Implements a 2D convolutional layer.
 */
class Conv2D : public Layer
{
public:
    Conv2D(Layer *prev, int n_kernels, int kernel_size, int stride,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Conv2D();
    size_t get_workspace_size() const override;
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    /** Descriptor for the shape of the convolutional filter to be learned */
    cudnnFilterDescriptor_t filter_desc;

    /** Descriptor for the biases */
    cudnnTensorDescriptor_t bias_desc;

    /** Descriptor for the actual convolution this layer will do */
    cudnnConvolutionDescriptor_t conv_desc;

    /** Convolutional algorithm to be used on forward pass */
    cudnnConvolutionFwdAlgo_t fwd_algo;

    /** Convolution algorithm to be used on backward pass wrt filters */
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

    /** Convolution algorithm to be used on backward pass wrt data */
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
};


/**
 * Implements a 2D pooling layer.
 */
class Pool2D : public Layer
{
public:
    Pool2D(Layer *prev, int stride, cudnnPoolingMode_t mode,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Pool2D();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    /** Descriptor of the pooling operation implemented by this layer */
    cudnnPoolingDescriptor_t pooling_desc;
};


/**
 * Transforms the final embedding of the neural network into a final prediction
 * in the output space. This prediction is to be used to compare against a true
 * output (stored in {\link Layer::grad_out_batch} and compute the loss and
 * accuracy of that prediction, relative to the ground truth.
 *
 * Subclasses of the Loss layer should set their output shape equal to the shape
 * of the true output.
 */
class Loss : public Layer {
public:
    Loss(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Loss();

protected:
    /** Loss incurred on the given data minibatch. */
    float loss = 0;

    /** Accuracy of prediction on the given data minibatch */
    float acc = 0;
};


/**
 * Implements a softmax activation and the categorical cross-entropy loss.
 */
class SoftmaxCrossEntropy : public Loss {
public:
    SoftmaxCrossEntropy(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~SoftmaxCrossEntropy();
    void forward_pass() override;
    void backward_pass(float lr) override;
    float get_loss() override;
    float get_accuracy() override;
};
