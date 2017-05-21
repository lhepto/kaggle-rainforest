from numpy import genfromtxt
import numpy as np
import io

# def prepsare_data():
#     #Generate Training Data from Files - Save to Numpy File
#     ylabels,y = get_labels(PLANET_KAGGLE_ROOT+"train.csv")
#     np.save(file=PLANET_KAGGLE_ROOT+"labelmatrix",arr=y)
#     np.save(file=PLANET_KAGGLE_ROOT+"labeldata",arr=ylabels)
#     pickle_image_data(y, ylabels, PLANET_KAGGLE_ROOT + "train-tif-v2/", PLANET_KAGGLE_ROOT + "train-pickles/")
#     exit()

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