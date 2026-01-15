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
images_path = os.path.join(path, "images", "Images") #address of 'Images' folder inside of the path
all_breeds = sorted(os.listdir(images_path))
num_classes = len(all_breeds)

# ---------- divide dataset into train, validate and test ----------
train_ds = tf.keras.utils.image_dataset_from_directory(
    images_path, # directory where all images are preserved
    validation_split=0.3, # 30% goes for vlidation
    subset="training", # either train or validation
    seed=420,# random mixing
    image_size=(224, 224),
    batch_size=32, # how many images in one batch
    class_names=all_breeds # the list of breeds which will be later used to code labels
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    images_path,
    validation_split=0.3,
    subset="validation",
    seed=420,
    image_size=(224, 224),
    batch_size=32,
    class_names=all_breeds
)

temp_batches = tf.data.experimental.cardinality(temp_ds) #we get all batches from the temp dataset
val_size = temp_batches // 2

val_ds = temp_ds.take(val_size) #create new dataset which consists of val_size batcehs
test_ds = temp_ds.skip(val_size)

# ---------- model creation ----------

# we take the pre-trained ResNet50 model
base = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224,224,3),
    weights='imagenet'
)
base.trainable = False


preprocess = tf.keras.applications.resnet.preprocess_input
preprocess_layer = layers.Lambda(preprocess)

model = tf.keras.Sequential([
    layers.Input((224,224,3)),
    preprocess_layer, 
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# ---------- compilation ----------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- training ----------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
)

# ---------- testing ----------
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)