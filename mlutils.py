import lasagne
import scipy
from sklearn.metrics import fbeta_score
import numpy as np
from random import randint
from skimage.transform import rescale,resize

# ############################# Batch iterator ###############################
def iterate_minibatches(ymatrix, ylabels, picklesdir, batchsize, shuffle=False, center = True, scale = True, rotate=True, resizeTo=0):
    assert len(ylabels) == len(ymatrix)

    if shuffle:
        indices = np.arange(len(ymatrix))
        np.random.shuffle(indices)

    for start_idx in range(0, len(ymatrix) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if (resizeTo==0):
            image_data_minibatch = np.empty(shape=(len(ylabels[excerpt]),4,256,256),dtype=np.uint16)
        else:
            image_data_minibatch = np.empty(shape=(len(ylabels[excerpt]), 4, resizeTo, resizeTo), dtype=np.uint16)

        it = 0
        for (pickle_name,labels) in ylabels[excerpt]:
            image = np.load(picklesdir + "/" + pickle_name+".npy")
            image = np.swapaxes(image, 0, 2)

            if (rotate):
                image = scipy.ndimage.rotate(image, randint(0, 3) * 90)

            if (resizeTo != 0):
                image = scipy.misc.imresize(image,(resizeTo,resizeTo,4))

            image = np.swapaxes(image, 0, 2)

            image_data_minibatch[it, :, :, :] = image#scipy.misc.imresize(image, size=(4, 224, 224))
            it += 1

        if (center):
            color_channel_means = np.mean(np.mean(image_data_minibatch,axis=0),axis=0)
            image_data_minibatch = (image_data_minibatch - color_channel_means)

        if (scale):
            color_channel_stdevs = np.std(np.std(image_data_minibatch,axis=0),axis=0)
            image_data_minibatch = image_data_minibatch/color_channel_stdevs # SCALE R,G,B,IR to mean 0 and stddev 1

        #image_data_minibatch = image_data_minibatch[:,1:3,:,:]

        yield image_data_minibatch.astype('float32'), ymatrix[excerpt]

# ############################# F2 Score ####################################
def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

# PICKLE THE MODEL TO FILE #################################################
def pickle_model_params(network):
    # Then we print the results for this epoch:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    print("pickled succesfully")

def depickle_model_params(network,path):
    with np.load(path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    return(network)