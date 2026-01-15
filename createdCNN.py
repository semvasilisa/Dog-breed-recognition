# ---------- imports ----------
import kagglehub # download data from kaggle
import os # pathnames, wokr with files and folders
import matplotlib.pyplot as plt # show image
from PIL import Image #reads jpg files

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras

# ---------- download dataset ----------
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
# print(path) #root folder of the dataset where all the data is preserved
# print(os.listdir(path)) #all files in path address
images_path = os.path.join(path, "images", "Images") #address of 'Images' folder inside of the path
# print("images path:",images_path)
# print("10 dog breeds:",os.listdir(images_path)[:10])
# dog_breeds = os.listdir(images_path)
# print("amount of the breeds:",len(dog_breeds)) #we have 120 breeds
# first_breed = os.path.join(images_path,dog_breeds[0]) #address of the first folder
# print(os.listdir(first_breed))


all_breeds = sorted(os.listdir(images_path))
selected_breeds = all_breeds[:10]

# ---------- print first dog ----------
# first_image_name = os.listdir(first_breed)[0]
# first_image_path = os.path.join(first_breed, first_image_name)
# img1 = Image.open(first_image_path)
# plt.imshow(img1)
# plt.axis('off')
# plt.show()
# print("dog breed is",dog_breeds[0])

# ---------- divide dataset into train, validate and test ----------
train_ds = tf.keras.utils.image_dataset_from_directory(
    images_path, # directory where all images are preserved
    validation_split=0.3, # 30% goes for vlidation
    subset="training", # either train or validation
    seed=420,# random mixing
    image_size=(224, 224),
    batch_size=32, # how many images in one batch
    class_names=selected_breeds # the list of breeds which will be later used to code labels
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    images_path,
    validation_split=0.3,
    subset="validation",
    seed=420,
    image_size=(224, 224),
    batch_size=32,
    class_names=selected_breeds
)

temp_batches = tf.data.experimental.cardinality(temp_ds) #we get all batches from the temp dataset
val_size = temp_batches // 2

val_ds = temp_ds.take(val_size) #create new dataset which consists of val_size batcehs
test_ds = temp_ds.skip(val_size)

# ---------- create normalization and augmentation layers ----------
normalization_layer = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2), # rotate on ±0.2rad
    tf.keras.layers.RandomZoom(0.3),# zoom ±30%
    tf.keras.layers.RandomContrast(0.3), # ±30%
    tf.keras.layers.RandomBrightness(0.2),# ±20%
])

# ---------- model creation ----------
model = models.Sequential()
model.add(layers.Input(shape=(224, 224, 3)))
#model.add(data_augmentation)
model.add(normalization_layer)
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',#adds nonlinearity for complex shape recognition
    #input_shape=(224, 224, 3)
    )
)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#before flatten (height, width, channels)
model.add(layers.Flatten())

#here dense combines all the patterns from previous layers
model.add(layers.Dense(256, activation='relu'))# here we have a vector of 128 numbers where each number is a combination of previous patterns
#model.add(layers.Dropout(0.5))#tun off 50% of neurons

#here dense softmax turns outputs to probabilities of the breeds
model.add(layers.Dense(10, activation='softmax'))

# ---------- compilation ----------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- training ----------

#model.fit() model teaching method, gives data to the model, calculates the loss, changes the weights, repeats the process
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10 #1 epoch = one full pass over the entire training dataset
)

