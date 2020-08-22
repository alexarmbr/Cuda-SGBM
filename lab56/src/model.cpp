/**
 * Layer-agnostic implementation of a neural network Model's core functionality
 * @author Aadyot Bhatnagar
 * @date April 22, 2018
 */

#include <string>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "layers.hpp"
#include "model.hpp"
#include "helper_cuda.h"

/**
 * Initializes a neural network that does a forwards and backwards pass on
 * minibatches of data. In each minibatch, there are n data points, each with
 * c channels, height h, and width w. For one-dimensional input data, set height
 * and width to 1, and let the number of channels c represent the dimensionality
 * of the input vectors.
 */
Model::Model(int n, int c, int h, int w) {
    this->has_loss = false;
    this->batch_size = n;
    this->input_size = c * h * w;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );
    CUDNN_CALL( cudnnCreate(&cudnnHandle) );
    this->layers = new std::vector<Layer *>;
    this->layers->push_back(new Input(n, c, h, w, cublasHandle, cudnnHandle));
}

Model::~Model() {
    std::vector<Layer *>::iterator it;
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
        delete *it;
    delete this->layers;
    if (workspace)
        CUDA_CALL( cudaFree(workspace) );
    CUBLAS_CALL( cublasDestroy(cublasHandle) );
    CUDNN_CALL( cudnnDestroy(cudnnHandle) );
}

/**
 * To add a layer to a model, one must specify the type of layer as a string, as
 * well as the arguments needed to instantiate such a layer. The arguments arg1,
 * arg2, and arg3 specify the parameters used in the constructor of the layer we
 * wish to initialize. They have default values of 0, and a user need only supply
 * the arguments necessary taken by the constructor for the desired layer.
 *
 * @param layer A name for the kind of layer we want to add to the model.
 * @param shape Parameters used to initialize the layer in question.
 */
void Model::add(std::string layer, std::vector<int> shape)
{
    assert(!this->has_loss && "Cannot add any layers after a loss function.");

    Layer *last = layers->back();
    std::transform(layer.begin(), layer.end(), layer.begin(), ::tolower);

    /* ReLU activation */
    if (layer == "relu")
    {
        layers->push_back(
            new Activation(last, CUDNN_ACTIVATION_RELU, 0.0,
                cublasHandle, cudnnHandle));
    }

    /* tanh activation */
    else if (layer == "tanh")
    {
        layers->push_back(
            new Activation(last, CUDNN_ACTIVATION_TANH, 0.0,
                cublasHandle, cudnnHandle));
    }

    /* Loss layers must also update that the model has a loss function */
    else if (layer == "softmax crossentropy" || layer == "softmax cross-entropy")
    {
        layers->push_back(new SoftmaxCrossEntropy(last, cublasHandle, cudnnHandle));
        this->has_loss = true;
    }

    /* Dense layer */
    else if (layer == "dense")
    {
        assert(!shape.empty() && shape[0] > 0 &&
            "Must specify positive output shape for dense layer.");
        layers->push_back(new Dense(last, shape[0], cublasHandle, cudnnHandle));
    }

    /* Convolutional layer */
    else if (layer == "conv")
    {
        assert(shape[0] > 0 &&
            "Must specify positive number of knernels for conv layer.");
        assert(shape[1] > 0 &&
            "Must specify positive kernel dimension for conv layer.");
        assert(shape[2] > 0 &&
            "Must specify positive stride for conv layer.");
        layers->push_back(
            new Conv2D(last, shape[0], shape[1], shape[2],
                cublasHandle, cudnnHandle));
    }

    /* Max pooling layer */
    else if (layer == "max pool")
    {
        assert(!shape.empty() && shape[0] > 0 &&
            "Must specify positive pooling dimension.");
        layers->push_back(
            new Pool2D(last, shape[0], CUDNN_POOLING_MAX,
                cublasHandle, cudnnHandle) );
    }

    else if (layer == "avg pool" || layer == "average pool" ||
        layer == "mean pool")
    {
        assert(!shape.empty() && shape[0] > 0 &&
            "Must specify positive pooling dimension.");
        layers->push_back(
            new Pool2D(last, shape[0],
                CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                cublasHandle, cudnnHandle) );
    }


    else assert(false && "Invalid layer specification.");
}

/**
 * Sets up any cuDNN workspace to be used by this model.
 *
 * @pre The layers of this model have all been added
 */
void Model::init_workspace()
{
    assert(this->has_loss && "All layers of model must have been added!");

    // Get the largest workspace needed by any layer
    std::vector<Layer *>::iterator it;
    for (it = layers->begin(); it != layers->end(); ++it)
        workspace_size = std::max(workspace_size, (*it)->get_workspace_size());

    // If any need a nonzero-sized workspace, initialize one appropriately
    // and give it to each layer individually
    if (workspace_size > 0)
    {
        CUDA_CALL( cudaMalloc(&workspace, workspace_size) );
        for (it = layers->begin(); it != layers->end(); ++it)
            (*it)->set_workspace(workspace, workspace_size);
    }
}

/**
 * Trains the model on the training examples given at the specified learning
 * rate for the desired number of epochs.
 *
 * @param train_X The entire dataset that we wish to train our model on
 *
 * @param train_Y The ground truth labels assigned to each entry in train_X
 *
 * @param lr The learning rate (the coefficient to multipy the gradient by when
 *             doing gradient descent)
 *
 * @param n_examples The number of training examples contained in train_X
 *
 * @param n_epochs The number of iterations over which we want to accumulate
 *                   gradient updates from the entire dataset train_X
 */
void Model::train(const float *train_X, float *train_Y, float lr, int n_examples,
    int n_epochs)
{
    int in_size = get_output_batch_size(layers->front());
    int out_size = get_output_batch_size(layers->back());
    int n_batches = n_examples / batch_size;

    for (int i = 0; i < n_epochs; ++i)
    {
        std::cout << "Epoch " << i + 1 << std::endl;
        std::cout << "--------------------------------------------------------------";
        std::cout << std::endl;

        float acc = 0;
        float loss = 0;

        // Train on every complete batch
        for (long curr_batch = 0; curr_batch < n_batches; curr_batch++)
        {
            const float *curr_batch_X = train_X + curr_batch * in_size;
            float *curr_batch_Y = train_Y + curr_batch * out_size;
            train_on_batch(curr_batch_X, curr_batch_Y, lr);

            // Update training statistics for this minibatch
            acc += this->layers->back()->get_accuracy();
            loss += this->layers->back()->get_loss();
        }

        std::cout << "Loss: " << loss / n_batches;
        std::cout << ",\tAccuracy: " << acc / n_batches;
        std::cout << std::endl << std::endl;
    }
}


/**
 * Has a trained model return its predictions on a set of data.
 *
 * @param pred_X A set of data points that we want our model to make
 *                 predictions on
 *
 * @param n_examples The number of data points in pred_X
 *
 * @return An array of the model's predictions on each data point in pred_X
 */
float *Model::predict(const float *pred_X, int n_examples)
{
    // variables in which to get input and output shape
    int in_size = get_output_batch_size(layers->front());
    int out_size = get_output_batch_size(layers->back());

    /* Allocate array for predictions */
    float *pred_Y = new float[out_size * n_examples / batch_size];

    /* Predict on every complete batch */
    int n_batches = n_examples / batch_size;
    for (int curr_batch = 0; curr_batch < n_batches; ++curr_batch) {
        float *curr = predict_on_batch(pred_X + curr_batch * in_size);
        CUDA_CALL( cudaMemcpy(pred_Y + curr_batch * out_size, curr,
            out_size * sizeof(float), cudaMemcpyDeviceToHost) );
    }

    return pred_Y;
}


/**
 * Returns the model's predictive performance on a set of inputs, given their
 * ground truth labels.
 *
 * @param eval_X A set of data points we want to evaluate a trained model's
 *               predictive performance
 *
 * @param eval_Y The ground truth labels for each data point in eval_X
 *
 * @param n_examples The number of data points in eval_X
 *
 * @return The model's average predictive performance on data (eval_X, eval_Y)
 */
result *Model::evaluate(const float *eval_X, float *eval_Y, int n_examples)
{
    int in_size = get_output_batch_size(layers->front());
    int out_size = get_output_batch_size(layers->back());
    int n_batches = n_examples / batch_size;

    // Allocate array for predictions
    result *ret = new result;
    ret->predictions = new float[out_size * n_batches];
    ret->acc = 0;
    ret->loss = 0;

    // Predict on every complete batch
    for (int curr_batch = 0; curr_batch < n_batches; ++curr_batch) {
        const float *curr_batch_X = eval_X + curr_batch * in_size;
        float *curr_batch_Y = eval_Y + curr_batch * out_size;
        result *curr = evaluate_on_batch(curr_batch_X, curr_batch_Y);

        // Copy results from batch into accumulator for full sample
        CUDA_CALL( cudaMemcpy(ret->predictions + curr_batch * out_size,
            curr->predictions, out_size * sizeof(float), cudaMemcpyDeviceToHost) );
        ret->acc = (ret->acc * curr_batch + curr->acc) / (curr_batch + 1u);
        ret->loss = (ret->loss * curr_batch + curr->loss) / (curr_batch + 1u);
        delete curr;
    }

    std::cout << "Validation" << std::endl;
    std::cout << "----------------------------------------------------";
    std::cout << std::endl << "Loss: " << ret->loss;
    std::cout << ",\tAccuracy: " << ret->acc << std::endl << std::endl;
    return ret;
}



/**
 * Does one stochastic gradient update on the model using a minibatch of input
 * data batch_X with corresponding ground truth labels batch_Y.
 *
 * @param batch_X The starting point of the minibatch of data we wish to use to
 *                  perform the next stochastic gradient descent update
 *
 * @param batch_Y The starting point of the minibatch of ground truth labels
 *                  associated with batch_X
 *
 * @param lr The learning rate (coefficient by which we multiply the gradient
 *           when adding it to the current parameter values)
 */
void Model::train_on_batch(const float *batch_X, float *batch_Y, float lr)
{
    assert(this->has_loss && "Cannot train without a loss function.");

    // Copy input and output minibatches into the model's buffers
    copy_input_batch(batch_X);
    copy_output_batch(batch_Y);

    // Do a forward pass through every layer
    std::vector<Layer *>::iterator it;
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
        (*it)->forward_pass();

    // Do a backward pass through every layer
    std::vector<Layer *>::reverse_iterator rit;
    for (rit = this->layers->rbegin(); rit != this->layers->rend(); ++rit)
        (*rit)->backward_pass(lr);
}


/**
 * Returns the model's predictions on a minibatch of input data starting at
 * batch_X.
 *
 * @param batch_X A minibatch of input data we want our model to make
 *                  predictions on
 *
 * @return The model's predictions on batch_X
 */
float *Model::predict_on_batch(const float *batch_X) {

    // Copy input (batch_X) into input layer and nothing into output layer
    copy_input_batch(batch_X);

    // Do a forward pass through every layer
    std::vector<Layer*>::iterator it;
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
        (*it)->forward_pass();

    // The predictions are the output of the last layer
    return this->layers->back()->get_output_fwd();
}


/**
 * Evalutes the model's predictions on an input minibatch batch_X with
 * true outputs batch_Y.
 *
 * @param batch_X A minibatch of data we want to use to evaluate our trained
 *                  model's performance
 *
 * @param batch_Y The starting point of the minibatch of ground truth labels
 *                  associated with batch_X
 *
 * @return The model's average predictive performance on data minibatch
 *           (batch_X, batch_Y)
 */
result *Model::evaluate_on_batch(const float *batch_X, float *batch_Y) {
    assert(this->has_loss && "Cannot evaluate without a loss function.");

    // Making predictions does a forward pass
    result *ret = new result;
    ret->predictions = predict_on_batch(batch_X);

    // Copy the output batch to the loss layer so we can evaluate our
    // predictions (since we've already done a forward pass)
    copy_output_batch(batch_Y);
    ret->acc = this->layers->back()->get_accuracy();
    ret->loss = this->layers->back()->get_loss();
    return ret;
}

/**
 * Copies a minibatch of inputs on the host to the device buffer corresponding
 * to the outputs of the input layer.
 *
 * @param batch_X    A pointer to the start of the host buffer containing the
 *                    minibatch of inputs we wish to copy. The size (in bytes) of
 *                    the minibatch is inferred from the input layer.
 */
void Model::copy_input_batch(const float *batch_X)
{
    Layer *input = layers->front();
    int in_size = get_output_batch_size(input);
    CUDA_CALL( cudaMemcpyAsync(input->get_output_fwd(), batch_X,
        in_size * sizeof(float), cudaMemcpyHostToDevice) );
}

/**
 * Copies a minibatch of outputs on the host to the device buffer corresponding
 * to the gradient with respect to the output of the loss layer.
 *
 * @param batch_Y    A pointer to the start of the host buffer containing the
 *                    minibatch of outputs we wish to copy. The size (in bytes) of
 *                    the minibatch is inferred from the loss layer.
 */
void Model::copy_output_batch(const float *batch_Y)
{
    Layer *loss = layers->back();
    int out_size = get_output_batch_size(loss);
    CUDA_CALL( cudaMemcpyAsync(loss->get_input_bwd(), batch_Y,
        out_size * sizeof(float), cudaMemcpyHostToDevice) );
}

/**
 * Returns the number of floats in an output minibatch of the given layer.
 *
 * @param layer    The layer whose output size we wish to find
 *
 * @return The size (in number of floats) of the output of {@code layer}
 */
int Model::get_output_batch_size(Layer *layer) const
{
    // variables in which to get output shape
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    // Figure out size of output minibatch
    cudnnTensorDescriptor_t out_shape = layer->get_out_shape();
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride) );

    return n * c * h * w;
}
