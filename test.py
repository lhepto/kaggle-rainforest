
import sys
import time

import scipy
import theano
import theano.tensor as T
import lasagne
import numpy as np

from skimage import io
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from mlutils import pickle_model_params,iterate_minibatches,f2_score,depickle_model_params
from modelsDNN import larger_cnn,basic_cnn
#from models import larger_cnn,basic_cnn
from skimage import viewer

def main():

    num_epochs = 150
    batch_size = 100

    PLANET_KAGGLE_ROOT = "B:/rainforest-kaggle"
    PICKLE_DIR = "train-pickles"



    y = np.load(PLANET_KAGGLE_ROOT + "/" + "labelmatrix.npy")
    ylabels = np.load(PLANET_KAGGLE_ROOT + "/" + "labeldata.npy")

    print("Loaded labels...")

    ytrain = y[0:100,]
    ytrainlabels = ylabels[0:100,]

    ytest = y[0:1000,]
    ytestlabels = ylabels[0:1000,]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    network = vgg16(input_var)

    network = depickle_model_params(network,'C:/Users/Lewis_2/Desktop/Kaggle Rainforest - attempt 1 vgg style CNN/kaggle-rainforest\model.npz')


    print ("Built model...")

    # create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    # determine gradient update method
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=.9)

    # create update function
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # create testing loss function - as per training fn, but deterministic
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_objective = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
    test_loss = test_objective.mean()

    # create accuracy function - binary y/n decision
    test_acc = T.mean(T.eq([target_var],T.round_half_away_from_zero(test_prediction)))

    # compile test func, returns accuracy and loss given inputs / outputs
    test_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # compile predict funcm rounds results to nearest binary label
    predict_fn = theano.function([input_var], T.round_half_away_from_zero(test_prediction))

    # print layer parameters -----------------------------------------------------------------
    print ("Model layer breakdown: --------------------------")
    all_params = lasagne.layers.count_params(network)

    for layer in lasagne.layers.get_all_layers(network):
        print(str(type(layer)) + " " + str(lasagne.layers.count_params(layer)/1e6))

    print("All params (millions):"+str(all_params/1e6)+"-----")

    pickle_model_params(network)

    # start training ------------------------------------------------------------------------
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        start_time = time.time()
        train_loss = 0
        train_batches = 0
        train_f2 = 0

        for batch in iterate_minibatches(ytrain, ytrainlabels, PLANET_KAGGLE_ROOT + "/" + PICKLE_DIR, batch_size, shuffle=False,rotate=True):
            inputs, targets = batch

            train_batches += 1
            train_f2 += f2_score(targets, predict_fn(inputs))
            train_loss += train_fn(inputs, targets)
            print("TRAINING (minibatch) - loss: {!s:}".format(train_loss / train_batches))

        print("TRAINING (epoch) - loss: {!s:} f2: {!s:}".format(train_loss/train_batches,train_f2/train_batches))

        test_batches = 0
        test_f2 = 0
        test_acc = 0
        test_loss = 0

        for batch in iterate_minibatches(ytest,ytestlabels, PLANET_KAGGLE_ROOT + "/" + PICKLE_DIR,batch_size, shuffle=False):
            inputs, targets = batch

            test_batches += 1
            this_loss, this_acc = test_fn(inputs, targets)
            test_acc += this_acc
            test_loss += this_loss
            test_f2 += f2_score(targets,predict_fn(inputs))

            print("TEST (minibatch) - loss: {!s:} accuracy: {!s:} f2: {!s:}".format(test_loss / test_batches,
                                                                               test_acc / test_batches,
                                                                               test_f2 / test_batches))

        print("TEST (epoch) - loss: {!s:} accuracy: {!s:} f2: {!s:}".format(test_loss/test_batches, test_acc/test_batches, test_f2 / test_batches))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

        # Pickle model params every 5 iterations
        if (epoch % 5 == 0):
            pickle_model_params(network)

        # Write output to file
        with open("output.txt", "a") as trackerfile:
            trackerfile.write("TRAINING (epoch) - loss: {!s:} f2: {!s:}".format(train_loss/train_batches,train_f2/train_batches))
            trackerfile.write(" TEST (epoch) - loss: {!s:} accuracy: {!s:} f2: {!s:}".format(test_loss/test_batches, test_acc/test_batches, test_f2 / test_batches))
            trackerfile.write("\n")

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