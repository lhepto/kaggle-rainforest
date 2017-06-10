import lasagne
import lasagne.layers.dnn

def basic_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 4, 256, 256),
                                        input_var=input_var)

    network = lasagne.layers.dnn.Conv2DDNNLayer(
           network, num_filters=10, filter_size=(5, 5),
           nonlinearity=lasagne.nonlinearities.leaky_rectify,
           W=lasagne.init.GlorotUniform())
    network = lasagne.layers.dnn.MaxPool2DDNNLayer(network, pool_size=(2, 2))

    network = lasagne.layers.dnn.Conv2DDNNLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.MaxPool2DDNNLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=17,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return (network)

def vgg16(input_var=None):
    from lasagne.layers import InputLayer
    from lasagne.layers import DenseLayer
    from lasagne.layers import NonlinearityLayer
    from lasagne.layers import DropoutLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.nonlinearities import softmax

    net = {}
    net['input'] = InputLayer((None, 4, 256, 256),input_var=input_var)
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
        net['fc7_dropout'], num_units=1000, nonlinearity=lasagne.nonlinearities.sigmoid)

    return net['fc8']

def larger_cnn(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None, 4, 256, 256),
                                        input_var=input_var)

    network = lasagne.layers.dnn.Conv2DDNNLayer(
           network, num_filters=8, filter_size=(3, 3),
           nonlinearity=lasagne.nonlinearities.leaky_rectify,
           W=lasagne.init.GlorotUniform())

    network = lasagne.layers.dnn.Conv2DDNNLayer(
           network, num_filters=8, filter_size=(3, 3),
           nonlinearity=lasagne.nonlinearities.leaky_rectify,
           W=lasagne.init.GlorotUniform())

    network = lasagne.layers.dnn.MaxPool2DDNNLayer(network, pool_size=(2, 2))

    network = lasagne.layers.dnn.Conv2DDNNLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.Conv2DDNNLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.MaxPool2DDNNLayer(network, pool_size=(2, 2))

    network = lasagne.layers.dnn.Conv2DDNNLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.Conv2DDNNLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.Conv2DDNNLayer(
        network, num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.MaxPool2DDNNLayer(network, pool_size=(2, 2))

    network = lasagne.layers.dnn.Conv2DDNNLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.Conv2DDNNLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.Conv2DDNNLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.dnn.MaxPool2DDNNLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=17,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return network
