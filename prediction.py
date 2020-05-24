############################################
# Part 3 - Predictions
############################################

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os

success_normal = 0
success_pneumonia = 0
false_positive = 0
false_negative = 0
total_samples_normal = 0
total_samples_pneumonia = 0

# get model
base_dir = os.path.dirname(__file__)
model = load_model(os.path.join(base_dir, 'saved-models', 'cnn1590319711.h5'))

print('')
print('******************************PREDICTIONS******************************')
print('')

for filename in os.listdir(os.path.join(base_dir, 'chest_xray', 'val', 'NORMAL')):
    total_samples_normal = total_samples_normal + 1

    # pre processing image
    test_image = image.load_img(os.path.join(base_dir, 'chest_xray', 'val', 'NORMAL', filename), target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0) # this is necessary because precit method expect 4 dimensions, 3 for image, and one for batch number
    result = model.predict(test_image)

    # result
    if result[0][0] == 1:
        prediction = 'PNEUMONIA'
        false_positive = false_positive + 1
    else:
        prediction = 'NORMAL'
        success_normal = success_normal + 1

    print('NORMAL: ' + prediction)

for filename in os.listdir(os.path.join(base_dir, 'chest_xray', 'val', 'PNEUMONIA')):
    total_samples_pneumonia = total_samples_pneumonia + 1

    # pre processing image
    test_image = image.load_img(os.path.join(base_dir, 'chest_xray', 'val', 'PNEUMONIA', filename), target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0) # this is necessary because precit method expect 4 dimensions, 3 for image, and one for batch number
    result = model.predict(test_image)

    # result
    if result[0][0] == 1:
        prediction = 'PNEUMONIA'
        success_pneumonia = success_pneumonia + 1
    else:
        prediction = 'NORMAL'
        false_negative = false_negative + 1

    print('PNEUMONIA: ' + prediction)

total_samples = success_normal + success_pneumonia + false_negative + false_positive
accuracy = ((success_normal + success_pneumonia) / total_samples) * 100

# Summary
print('')
print('******************************SUMMARY******************************')
print('')
print('Total samples: ' + str(total_samples))
print('Accuracy: ' + str(accuracy) + '%')
print('')
print('------NORMAL------')
print('Total samples NORMAL: ' + str(total_samples_normal))
print('Prediction correct NORMAL: ' + str(success_normal))
print('False positives: ' + str(false_positive))
print('')
print('------PNEUMONIA------')
print('Total samples PNEUMONIA: ' + str(total_samples_pneumonia))
print('Prediction correct PNEUMONIA: ', str(success_pneumonia))
print('False negatives: ' + str(false_negative))