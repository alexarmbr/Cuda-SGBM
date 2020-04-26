/**
 * Builds and trains a neural network on the MNIST dataset of handwritten digits
 * @author Aadyot Bhatnagar
 * @date April 22, 2018
 */

#include <string>
#include <cstring>
#include <iostream>
#include "model.hpp"
#include "MNISTParser.h"

int main(int argc, char **argv)
{
    // Kind of activation to use (default relu)
    std::string activation = "relu";

    // Directory in which training and testing data are stored (default is this)
    std::string dirname = "/srv/cs179_mnist";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--dir") == 0 || strcmp(argv[i], "-d") == 0)
        {
            i++;
            if (i < argc)
                dirname = argv[i];
        }

        else if (strcmp(argv[i], "--act") == 0 || strcmp(argv[i], "-a") == 0)
        {
            i++;
            if (i < argc)
                activation = argv[i];
        }
    }

    // Load training set
    int n_train, c, h, w, n_classes;
    float *train_X, *train_Y;
    LoadMNISTData(dirname + "/train-images.idx3-ubyte",
        dirname + "/train-labels.idx1-ubyte",
        n_train, c, h, w, n_classes, &train_X, &train_Y);
    std::cout << "Loaded training set." << std::endl;

    // Initialize a model to classify the MNIST dataset
    Model *model = new Model(200, c, h, w);
#if CONV
    model->add("conv", { 20, 5, 1 });
    model->add("max pool", { 2 });
    model->add(activation);
    model->add("conv", { 5, 3, 1 });
    model->add("max pool", { 2 });
#else
    model->add("dense", { 200 });
#endif
    model->add(activation);
    model->add("dense", { n_classes });
    model->add("softmax crossentropy");
    model->init_workspace();

    // Train the model on the training set for 25 epochs
    std::cout << "Predicting on " << n_classes << " classes." << std::endl;
    model->train(train_X, train_Y, 0.03f, n_train, 25);

    // Load test set
    int n_test;
    float *test_X, *test_Y;
    LoadMNISTData(dirname + "/test-images.idx3-ubyte",
        dirname + "/test-labels.idx1-ubyte",
        n_test, c, h, w, n_classes, &test_X, &test_Y);
    std::cout << "Loaded test set." << std::endl;

    // Evaluate model on the test set
    result *res = model->evaluate(test_X, test_Y, n_test);

    // Delete all dynamically allocated data
    delete[] res->predictions;
    delete res;
    delete model;
    delete[] train_X;
    delete[] train_Y;
    delete[] test_X;
    delete[] test_Y;

    return 0;
}
