import numpy as np
from skimage import io
from os import walk
import csv
import sys
import sklearn.preprocessing as pp
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne

# Data operations
def get_labels(no_images, path):
    # Parse the training data into an nparray
    training_data = np.empty((no_images), dtype=object)
    row_iterator = 0
    with open(path, 'rt', encoding="UTF8") as f:
        reader = csv.reader(f)
        for row in reader:
            if (row_iterator > 0) and (row_iterator <= no_images):
                training_data[row_iterator - 1] = row[1].split(" ")
            row_iterator += 1

    # Find all the labels
    #all_labels = np.unique(
    #    np.asarray(([item for sublist in training_data for item in sublist])));  # flattens all label data

    all_labels = np.load("labels.npy")

    # Build a large matrix IMAGES X TAG CODE
    training_data_matrix = np.zeros((no_images, all_labels.shape[0]), dtype=np.uint16)
    row_iterator = 0
    with open(path, 'rt', encoding="UTF8") as f:
        reader = csv.reader(f)
        for row in reader:
            if (row_iterator > 0) and (row_iterator <= no_images):
                training_data[row_iterator - 1] = row[1].split(" ")
                for col in training_data[row_iterator - 1]:
                    training_data_matrix[(row_iterator - 1, np.where(all_labels == col))] = 1
            row_iterator += 1

    return (training_data_matrix)

def get_image_data(no_image, directory):
    # Extract all training images from a directory
    training_images = []
    for (dirpath, dirnames, filenames) in walk(directory):
        training_images.extend(filenames)
        break

    # Use skimage to put all the training data into a big array which is IMAGES,LEFT,RIGHT,CHANNEL
    training_image_data = np.zeros((no_image, 256, 256, 4), dtype=np.double)
    image_iterator = 0
    for training_image in training_images:
        if (image_iterator < no_image):
            raw_data = (io.imread(directory+training_image))
            color_channel_means = np.mean(np.mean(raw_data,axis=0),axis=0)
            color_channel_stdevs = np.std(np.std(raw_data,axis=0),axis=0)

            raw_data = (raw_data - color_channel_means)/color_channel_stdevs # SCALE R,G,B,IR to mean 0 and stddev 1

            training_image_data[image_iterator, :, :, :] = raw_data
            image_iterator += 1

    training_image_data = np.swapaxes(training_image_data,1,3)

    return(training_image_data)



def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 4, 256, 256),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
           network, num_filters=10, filter_size=(5, 5),
           nonlinearity=lasagne.nonlinearities.leaky_rectify,
           W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)



    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))



    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=17,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

0# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main():

    train_images = 200
    test_images = 100
    num_epochs = 20

    PLANET_KAGGLE_ROOT = "B:/rainforest-kaggle/"
    TRAIN_SET = "lewis-data/mini/train/"
    TEST_SET = "lewis-data/mini/test/"

    # Generate Training Data from Files - Save to Numpy File
    y_train = get_labels(train_images,PLANET_KAGGLE_ROOT+"lewis-data/mini/train.csv")
    X_train = get_image_data(train_images,PLANET_KAGGLE_ROOT+TRAIN_SET)

    y_test = get_labels(test_images,PLANET_KAGGLE_ROOT+"lewis-data/mini/test.csv")
    X_test = get_image_data(test_images,PLANET_KAGGLE_ROOT+TEST_SET)

    np.save("training_image_data",X_train)
    np.save("training_labels_data",y_train)

    np.save("test_image_data",X_test)
    np.save("test_labels_data",y_test)

    # Quick load the data
    X_train = np.load("training_image_data.npy")
    y_train = np.load("training_labels_data.npy")

    X_test = np.load("test_image_data.npy")
    y_test = np.load("test_labels_data.npy")

    print("loaded data")

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.imatrix('targets')

    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=0.001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_objective = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
    test_loss = test_objective.mean()

    # Also create an expression for the classification accuracy:

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    training_acc = T.mean(T.eq(T.round_half_away_from_zero(test_prediction), target_var), dtype=theano.config.floatX)
    training_fn = theano.function([input_var, target_var], [test_loss, training_acc])

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 200, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

         # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        testErr, testAcc = training_fn(X_test, y_test)
        trainingErr, trainingAcc = training_fn(X_train, y_train)

        print("TRAINING: loss: \t\t{!s:} accuracy: {!s:}".format(trainingErr, trainingAcc))
        print("TEST: loss: \t\t{!s:} accuracy: {!s:}".format(testErr, testAcc))

    print(predict_fn(X_train))







# Booting code
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)