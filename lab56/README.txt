This is a library that can be used to implement a fully connected neural network
for classification purposes. A high level interface for constructing a linear
feed-forward network is implemented in model.hpp and model.cpp. To add a layer,
a user should call the add() method of the Model and supply as arguments a
string indicating the type of layer desired, and an array or vector of integers
that specify parameters relevant to the layer. Note that the shape of a single
input minibatch (Number of examples in minibatch, Channels per example,
Height of example, Width of example) must be given to the Model's constructor
as NCHW.

Once a Model is constructed, a user can call the Model's train() method with
a set of training data of the specified input shape, a set of corresponding
labels with the specified output shape, the number of training examples being
used, and the desired mini-batch size. Similar functions exist to return the
Model's predictions (predict()) on a test set and to evaluate the Model's
performance on a test set, given a set of ground truth labels (evaluate()). An
example with a simple Model for Gaussian discrimination is included in main.cpp.

The Model class provides an abstraction layer  over a vector of generic Layer
objects. Each Layer (implementations in layers.hpp and layers.cpp) is a subclass 
of an abstract Layer superclass and implements its own methods for a forward and
backward pass on a minibatch of data. Notably, loss layers are implemented as
subclasses of an abstract Loss class, which is in turn a subclass of the
abstract Layer class. To add a new Layer, one need only implement an appropriate
class in layers.hpp and layers.cpp (see below for more details) and modify
the Model's add() method to parse arguments specifying the desired layer and its
parameters. 

Each Layer maintains device pointers to its internal representation of (1) its
input and output minibatches, (2) the gradients of the loss function with
respect to them, and (3) the current weights and biases associated with that
layer, and (4) the gradients of the weights and biases. Notably, the pointer to
a layer's input minibatch (and its gradient) is the same as the pointer to the
previous layer's output minibatch (and its gradient). See layers.hpp for more
details.

In its constructor, every subclass of the Layer base class must (1) set the
shape of its output minibatch of data, (2) set descriptors for other relevant
quantities/characteristics of the layer (e.g. shape of weight matrix, descriptor
of activation, descriptor of convolution operation, etc.) and (3) allocate
buffers for weights, biases, and output minibatches.

Each Layer subclass must also specify how it interacts with a feedforward
neural net by overriding the pure virtual methods Layer::forward_pass() (to do
its computation and pass it to the next layer) and Layer::backward_pass() (to
compute gradients and propagate them backwards to the previous layer). This
design makes it simple to add support for new kinds layers.

Implementations for new layers should be split between layers.hpp/layers.cpp,
while any GPU utility functions used for those layers (not CPU-callable CUDA
libraries, but actual kernels and the wrappers used to call them from the CPU)
should be placed in utils.cuh/utils.cu.