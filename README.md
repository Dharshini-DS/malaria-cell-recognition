# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset

Using data augmentation in the Convolutional Neural Network approach decreases the chances of overfitting. Thus, Malaria detection systems using deep learning proved to be faster than most of the traditional techniques. A Convolutional Neural Network was developed and trained to classify between the parasitized and uninfected smear blood cell images. The classical image features are extracted by CNN which can extract theimage features in three different categories – low-level, mid-level, and high-level features.

## Neural Network Model

![1](https://github.com/Dharshini-DS/malaria-cell-recognition/assets/93427345/a282a20d-23c3-48f5-8b7e-ecf2e7fa2b42)

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Read the dataset

### STEP 3:
Create an ImageDataGenerator to flow image data

### STEP 4:
Build the convolutional neural network model and train the model

### STEP 5:
Fit the model

### STEP 6:
Evaluate the model with the testing data

### STEP 7:
Fit the model

### STEP 8:
Plot the performance plot

## PROGRAM

```
Developed BY : Dharshini DS
Reg no : 212221230022
```
### Importing Modules
```
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
```
### DataDirectory
```
my_data_dir = '/home/ailab/hdd/dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir + '/test/'
train_path = my_data_dir + '/train/'
os.listdir(train_path)
len(os.listdir(train_path + '/uninfected/'))
len(os.listdir(train_path + '/parasitized/'))
os.listdir(train_path + '/parasitized')[0]
```
### Import and display an image
```
para_img = imread(train_path + '/parasitized/' + os.listdir(train_path + '/parasitized')[0])
para_img.shape
plt.imshow(para_img)
```
### Checking the image dimensions
```
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path + '/uninfected'):
    img = imread(test_path + '/uninfected' + '/' + image_filename)
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1, y=dim2)
image_shape = (130, 130, 3)
image_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    rescale=1/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
```
### Create a neural network model
```
model = models.Sequential([
    layers.Input((130, 130, 3)),
    layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", metrics='accuracy', optimizer="adam")
model.summary()
```
### Data generation for training and testing
```
train_image_gen = image_gen.flow_from_directory(train_path, target_size=image_shape[:2], color_mode='rgb',
                                                batch_size=16, class_mode='binary')

train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path, target_size=image_shape[:2], color_mode='rgb',
                                              batch_size=16, class_mode='binary', shuffle=False)

train_image_gen.class_indices
```
### Train the model
```
results = model.fit(train_image_gen, epochs=5, validation_data=test_image_gen)
model.save('cell_model1.h5')
```
### Visualize the training losses
```
losses = pd.DataFrame(model.history.history)
losses.plot()
```
### Evaluate the model
```
model.evaluate(test_image_gen)
```
### Make predictions and evaluate the model
```
pred_probabilities = model.predict(test_image_gen)
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes, predictions))
confusion_matrix(test_image_gen.classes, predictions)
```
### Load and process a single image for prediction
```
from tensorflow.keras.preprocessing import image

img = image.load_img('new.png')
img = tf.convert_to_tensor(np.asarray(img))
img = tf.image.resize(img, (130, 130))
img = img.numpy()

type(img)
plt.imshow(img)

x_single_prediction = bool(model.predict(img.reshape(1, 130, 130, 3)) > 0.6)
print(x_single_prediction)

if x_single_prediction == 1:
    print("Uninfected")
else:
    print("Parasitized")
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![4-1](https://github.com/Dharshini-DS/malaria-cell-recognition/assets/93427345/ee30d26d-658b-44c2-a3a8-066fee54c420)

### Classification Report

![4-2](https://github.com/Dharshini-DS/malaria-cell-recognition/assets/93427345/7e99aaa2-1a5c-4180-a9e1-97c4982070b6)

### Confusion Matrix

![4-3](https://github.com/Dharshini-DS/malaria-cell-recognition/assets/93427345/d62dd75f-4f2d-4e47-a321-52bb14c5f04b)

### New Sample Data Prediction

![4-4](https://github.com/Dharshini-DS/malaria-cell-recognition/assets/93427345/1ebdcb5c-2590-484c-a3ca-5446545ee125)

## RESULT
Thus, a deep neural network for Malaria infected cell recognized and analyzed the performance.
