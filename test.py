from skimage import io
import sys
import time
import theano
import theano.tensor as T
import lasagne
from numpy import genfromtxt
from sklearn.metrics import fbeta_score
import numpy as np

# Data operations
def get_labels(path):

    label_data = genfromtxt(path, delimiter=',',dtype=np.str)[1:,:] # read csv, remove header row
    all_labels = np.load("labels.npy") # read labels

    # Build a large matrix IMAGES X TAG CODE
    training_data_matrix = np.zeros((len(label_data), all_labels.shape[0]), dtype=np.uint16)
    row_iterator = 0

    for row in label_data:
        split_labels = row[1].split(" ")
        for col in split_labels:
            training_data_matrix[row_iterator,np.where(all_labels == col)] = 1
        row_iterator += 1

    # Test is in line
    assert (min(label_data[4,1].split(" ") == all_labels[training_data_matrix[4,:].astype(bool)])== True)
    assert (min(label_data[40, 1].split(" ") == all_labels[training_data_matrix[40, :].astype(bool)]) == True)

    return (label_data,training_data_matrix)

def pickle_image_data(y, ylabels, imageDirectory, pickleDirectory):
    # Extract all training images from a directory
    # Use skimage to put all the training data into a big array which is IMAGES,LEFT,RIGHT,CHANNEL

    iterator = 0

    for (image_name,labels) in ylabels:

            raw_data = io.imread(imageDirectory+image_name+".tif")
            np.save(file=pickleDirectory + "train_" + str(iterator), arr=np.swapaxes(raw_data,0,2))

            iterator += 1

            if (iterator % 500 == 0):
                print("image" + str(iterator))

def basic_cnn(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None, 4, 256, 256),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
           network, num_filters=10, filter_size=(5, 5),
           nonlinearity=lasagne.nonlinearities.leaky_rectify,
           W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=17,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

def vgg16(input_var=None):
    from lasagne.layers import InputLayer
    from lasagne.layers import DenseLayer
    from lasagne.layers import DropoutLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer

    def build_model():
        net = {}
        net['input'] = InputLayer((None, 4, 256, 256))
        net['conv1_1'] = ConvLayer(
            net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(
            net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(
            net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(
            net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(
            net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(
            net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(
            net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(
            net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(
            net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(
            net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(
            net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(
            net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(
            net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(
            net['fc7_dropout'], num_units=17, nonlinearity=lasagne.nonlinearities.sigmoid)

        return net['fc7_dropout']

# ############################# Batch iterator ###############################
def iterate_minibatches(ymatrix, ylabels, picklesdir, batchsize, shuffle=False, center = True, scale = True):
    assert len(ylabels) == len(ymatrix)

    if shuffle:
        indices = np.arange(len(ymatrix))
        np.random.shuffle(indices)

    for start_idx in range(0, len(ymatrix) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        image_data_minibatch = np.empty(shape=(len(ylabels[excerpt]),4,256,256),dtype=np.uint16)

        it = 0
        for (pickle_name,labels) in ylabels[excerpt]:
            image_data_minibatch[it,:,:,:] = np.load(picklesdir + pickle_name+".npy")
            it += 1

        if (center):
            color_channel_means = np.mean(np.mean(image_data_minibatch,axis=0),axis=0)
            image_data_minibatch = (image_data_minibatch - color_channel_means)

        if (scale):
            color_channel_stdevs = np.std(np.std(image_data_minibatch,axis=0),axis=0)
            image_data_minibatch = image_data_minibatch/color_channel_stdevs # SCALE R,G,B,IR to mean 0 and stddev 1

        yield image_data_minibatch,ymatrix[excerpt]

# ############################# F2 Score ####################################
def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def main():

    num_epochs = 20

    PLANET_KAGGLE_ROOT = "B:/rainforest-kaggle/"
    PICKLE_DIR = "train-pickles/"

    # Generate Training Data from Files - Save to Numpy File
    #ylabels,y = get_labels(PLANET_KAGGLE_ROOT+"train.csv")
    #np.save(file=PLANET_KAGGLE_ROOT+"labelmatrix",arr=y)
    #np.save(file=PLANET_KAGGLE_ROOT+"labeldata",arr=ylabels)
    #pickle_image_data(y, ylabels, PLANET_KAGGLE_ROOT + "train-tif-v2/", PLANET_KAGGLE_ROOT + "train-pickles/")
    #exit()

    y = np.load(PLANET_KAGGLE_ROOT+"labelmatrix.npy")
    ylabels = np.load(PLANET_KAGGLE_ROOT+"labeldata.npy")

    print("Loaded labels...")

    ytrain = y[0:800,]
    ytrainlabels = ylabels[0:800,]

    ytest = y[0:800,]
    ytestlabels = ylabels[0:800,]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    network = basic_cnn(input_var)

    print ("Built model...")

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

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    # training_acc = T.mean(T.eq(T.round_half_away_from_zero(test_prediction), target_var), dtype=theano.config.floatX)

    predict_fn = theano.function([input_var],T.round_half_away_from_zero(test_prediction))

    # all_layer_params = lasagne.layers.get_all_param_values(network)
    #
    # all_params = 0
    # for layer in all_layer_params:
    #     this_params = np.prod(layer.shape)
    #     all_params += this_params
    #     print("layer params:"+this_params)

    #print("all params:"+all_params/1e6)

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        train_f2 = 0
        for batch in iterate_minibatches(ytrain, ytrainlabels, PLANET_KAGGLE_ROOT + PICKLE_DIR, 200, shuffle=False):
            inputs, targets = batch

            train_err += train_fn(inputs, targets)
            train_batches += 1
            train_f2 += f2_score(targets,predict_fn(inputs))
            print("train minibatch")

         # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("TRAINING: loss: \t\t{!s:} accuracy: {!s:} f2: {!s:}".format(train_err/train_batches,0,train_f2/train_batches))

        test_f2 = 0
        test_batches = 0
        for batch in iterate_minibatches(ytest,ytestlabels,PLANET_KAGGLE_ROOT + PICKLE_DIR,100, shuffle=False):
            inputs, targets = batch

            test_batches += 1
            test_f2 += f2_score(targets,predict_fn(inputs))
            print("test minibatch")

        print("TEST: loss: \t\t{!s:} accuracy: {!s:} f2: {!s:}".format("", "",test_f2 / test_batches))







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