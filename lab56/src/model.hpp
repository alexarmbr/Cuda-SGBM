/**
 * Header file defining a Model, the class that represents a neural network
 * @author Aadyot Bhatnagar
 * @date April 22, 2018
 */


#pragma once

#include <vector>
#include <cublas_v2.h>
#include <cudnn.h>
#include "layers.hpp"

/** A struct to hold the results returned from evaluating a model */
typedef struct _result {
    float loss;
    float acc;
    float *predictions;
} result;

/**
 * A Model acts as a wrapper for the sequence of layers that comprise a feed-
 * forward neural network. The sequence of layers is stored in a class field
 * as a std::vector.
 */
class Model {
public:
    Model(int n, int c, int h = 1, int w = 1);
    ~Model();

    void add(std::string layer_type, std::vector<int> shape = {});
    void init_workspace();

    void train(const float *train_X, float *train_Y, float lr,
        int num_examples, int n_epochs);
    float *predict(const float *pred_X, int num_examples);
    result *evaluate(const float *eval_X, float *eval_Y, int num_examples);

private:
    void train_on_batch(const float *batch_X, float *batch_Y, float lr);
    float *predict_on_batch(const float *batch_X);
    result *evaluate_on_batch(const float *batch_X, float *batch_Y);

    void copy_input_batch(const float *batch_X);
    void copy_output_batch(const float *batch_Y);

    int get_output_batch_size(Layer *layer) const;

    /** Whether a loss layer has been added */
    bool has_loss;

    /** The number of examples in each minibatch */
    int batch_size;

    /** The size (in floats) of a single input */
    int input_size = 0;

    /** A vector containing the layers this neural net is comprised of */
    std::vector<Layer *> *layers;

    /** cuBLAS library context */
    cublasHandle_t cublasHandle;

    /** cuDNN library context */
    cudnnHandle_t cudnnHandle;

    /** Workspace for cuDNN scratch work during convolutions */
    float *workspace = nullptr;

    /** Size of {\link Model::workspace} in bytes */
    size_t workspace_size = 0;
};
