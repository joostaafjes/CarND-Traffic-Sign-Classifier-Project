# Load pickled data
import pickle

import pandas as pd
import numpy as np
import csv
import matplotlib.image as mpimg

# Fill this in based on where you saved the training and testing data
training_file = "./data/01-data-from-udacity/train.p"
validation_file = "./data/01-data-from-udacity/valid.p"
testing_file = "./data/01-data-from-udacity/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# The pickled data is a dictionary with 4 key/value pairs:
#
# 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
# 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
# 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

#  Number of training examples
n_train = np.shape(y_train)[0]

# Number of validation examples
n_validation = np.shape(y_valid)[0]

# Number of testing examples.
n_test = np.shape(y_test)[0]

# What's the shape of an traffic sign image?
image_shape = [np.shape(X_train)[1], np.shape(X_train)[2]]

# How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).size

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
# %matplotlib inline
plt.ion()
plt.interactive(True)

f, grid = plt.subplots(6, 8)
plt.subplots_adjust(hspace=0.1, wspace=0.1)

with open('signnames.csv', 'r') as signnames:
    sign_reader = csv.reader(signnames)
    for sign in sign_reader:
        cat_id = str(sign[0])
        if cat_id.isnumeric():
            # get meta data for cat
            cat_id = int(cat_id)
            desc = sign[1]
            index = np.where(y_train == cat_id)
            size = index[0].size
            print("{0}({1}): {2}".format(cat_id, size, desc))

            # get first image from training set
            first_image = X_train[index[0][0]]
            first_image = np.true_divide(first_image, 255)

            # show image
            #plt.subplot(8, 6, cat_id + 1)
            # plt.imshow(first_image)
            #plt.title("{0}({1}): {2}".format(cat_id, size, desc))
            # plt.show()
            grid[int(cat_id / 8), cat_id % 8].imshow(first_image)
    plt.show()





