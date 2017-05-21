import lasagne

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

    return (network)

def larger_cnn(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None, 4, 256, 256),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
           network, num_filters=8, filter_size=(3, 3),
           nonlinearity=lasagne.nonlinearities.leaky_rectify,
           W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
           network, num_filters=8, filter_size=(3, 3),
           nonlinearity=lasagne.nonlinearities.leaky_rectify,
           W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(3, 3),
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
