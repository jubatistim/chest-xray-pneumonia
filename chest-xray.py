# Convolutional Neural Network

### Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import os
import calendar
import time

###############################
# Part 1 - Building CNN
###############################

# Initializing CNN
classifier = Sequential()

# Step 1 - Convolution
# 3 first parameters
#       nb_filters = number of filters (feature filter to create feature map) for convolution - tipically starts with 32, and then add another convolution layers with more filters (64, 128, 256...)
#       nb_rows = rows of filter
#       nb_columns = columns of filter
#       input_shape = image size + number of channels (WARNING: for Theano backend use (128, 128, 3)
#       activation = activation function for the output of current layer
classifier.add(Convolution2D(32, 6, 6, input_shape=(64, 64, 3), activation = 'relu'))

# Step 2 - Max Pooling
#       pool_size = shape of the pooling vector, recommended is 2x2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 1 and 2 one more time - THIS IS DEEP LEARNING!!
classifier.add(Convolution2D(64, 6, 6, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
#       units = experimental value, something not so high to not cost much of computation, but not so low to not power the model. Usually is a power of 2. Or, of course is the number of categories we need for output
#       activation = activation function for the output of current layer
classifier.add(Dense(activation = 'relu', units = 128))
classifier.add(Dense(activation = 'sigmoid', units = 1))

# Compiling model
#       optimizer = algorithm used to calculate teh weights, usually is adam
#       loss = function used to validate the errors and improve the search for the minor errors, binary_crossentropy is for binary outputs, if we have more use categorical_crossentropy
#       metrics = metrics used to evaluate model during the training, usually accuracy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

############################################
# Part 2 - Fitting CNN to the images
############################################

from keras.preprocessing.image import ImageDataGenerator

# Image Augmentation
# Consider random small changes in trainning dataset to improve the capacity of generalization of the model, this object is just for prepare the train data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

# Image Augmentation
# for test is not necessary changes
test_datagen = ImageDataGenerator(rescale=1./255)

# Image Augmentation and prepare training data
# Training set
#       target_size should be the size of input of first convolution layer
#       batch_size is the size of packages presented to the model at once within an epoch, the number of batches presented to the model will be total samples divided by batch_size
#       class_mode define basically if we are working with binary classification or categorical
training_set = train_datagen.flow_from_directory('chest_xray/train', target_size=(64, 64), batch_size=32, class_mode='binary')

# Image Augmentation and prepare training data
# Test set - the parameters are the same as above
test_set = test_datagen.flow_from_directory('chest_xray/test', target_size=(64, 64), batch_size=32, class_mode='binary')

# !!!Train the classifier!!!
#       samples_per_epoch is the total number of samples in the training dataset
#       nb_val_samples is the total number of samples in the test dataset
#       nb_epoch is the number of times all samples will be shown to the model for weights adjustments
classifier.fit_generator(training_set, samples_per_epoch = 5216, nb_epoch=100, validation_data=test_set, nb_val_samples = 624)

# Save
name_tosave = 'cnn' + str(calendar.timegm(time.gmtime()))

# Save the class indices
file1 = open(os.path.join('./saved-models', name_tosave + '.txt'),"w")
file1.writelines(str(training_set.class_indices)) 
file1.close() #to change file access modes

# Save the model
classifier.save(os.path.join('./saved-models', name_tosave + '.h5'))

############################################
# Part 3 - Predictions
############################################

# Check file prediction.py
