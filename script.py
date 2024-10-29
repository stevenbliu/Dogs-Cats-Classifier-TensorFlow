# %% [markdown]
# Turning Images into arrays

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# %%
data_dir = "C:/Users/Steven/Desktop/TensorFlow/Data/PetImages"
categories = ["Dog", "Cat"]

for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

# %%
# Changing Pixels

img_size = 120
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap='gray')
plt.show()

# %%
training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# %%
print(len(training_data))

# %%
import random

random.shuffle(training_data)

# %%
for sample in training_data[:10]:
    print(sample[1])

# %%
X = []
y = []

# %%
for features, label in training_data:
    X.append(features)
    y.append(label)

# reshape (-1, IMG_SIZE, IMG_SIZE, number of colors)
X = np.array(X).reshape(-1, img_size, img_size, 1)

# %%
# Saving your data
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# %% [markdown]
# Use CNN to make a classifier with images

# %%
# Convulational Neural Networks

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Load Saved Data
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

#X = tf.keras.utils.normalize(X, axis=1)
X = X/255.0

# %%

model = Sequential()
model.add(Conv2D((64), (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D((64), (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

model.fit(X, np.array(y), batch_size=32, epochs=3, validation_split=.1)

# %% [markdown]
# 75% Accuracy is a good classification model for these images, and we can expect to get better as we increase the number of epochs.

# %%


# %%


# %%



