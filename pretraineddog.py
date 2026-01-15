# ---------- imports ----------
import kagglehub # download data from kaggle
import os # pathnames, work with files and folders
import matplotlib.pyplot as plt # show image
from PIL import Image # reads jpg files

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras

import numpy as np
from tensorflow.keras.preprocessing import image

import cv2
from collections import Counter


# ---------- download dataset ----------

path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
images_path = os.path.join(path, "images", "Images") #address of 'Images' folder inside of the path
all_breeds = sorted(os.listdir(images_path))
num_classes = len(all_breeds)
print(num_classes)  

# ---------- divide dataset into train, validate and test ----------

train_ds = tf.keras.utils.image_dataset_from_directory(
    images_path, # directory where all images are preserved
    validation_split=0.3, # 30% goes for validation
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

# we divide temporary dataset into validation and testing datasets
temp_batches = tf.data.experimental.cardinality(temp_ds) # we get all batches from the temp dataset
val_size = temp_batches // 2

val_ds = temp_ds.take(val_size) # create new dataset which consists of val_size batcehs
test_ds = temp_ds.skip(val_size) # val_ds and test_ds have the same size

# ---------- model creation ----------

# returns a map of features (7x7x1280) where 1280 are recognised patterns
base = tf.keras.applications.EfficientNetB0(
    include_top=False, # remove classification layer, because we need to change it for dog breed recognition
    input_shape=(224, 224, 3), # our input images size
    weights='imagenet' # get the weights EfficientNetB0 have from learning on imagenet dataset 
  )
base.trainable = False # freeze knowledge, weights will not change
base.training = False # evaluation mode


preprocess = tf.keras.applications.efficientnet.preprocess_input # change image extension from [0,255] to [-1,1]
preprocess_layer = layers.Lambda(preprocess) # turn 'processing' into "normalization layer"

model = tf.keras.Sequential([
    layers.Input((224,224,3)), # input image size
    preprocess_layer, # normalization for EfficientNet
    base, # pretrained model
    tf.keras.layers.GlobalAveragePooling2D(),# turns matrix into vector with 1280 recognised patterns
    tf.keras.layers.Dropout(0.4), # set to zero 40% of random patterns
    tf.keras.layers.Dense(256, activation='relu'), # combine patterns
    tf.keras.layers.Dropout(0.3), # set to zero 30% of random patterns
    tf.keras.layers.Dense(num_classes, activation='softmax') # give set of predictions for each breed
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

# ---------- fine-tuning ----------

# base.trainable = True

# fine_tune_at = int(len(base.layers) * 0.7) # how many layers is in 70% of the EfficientNet

# #don't train layers from [0-fine_tune_at]
# for layer in base.layers[:fine_tune_at]:
#     layer.trainable = False

# model.compile(
#     # we need to choose slow learning rate in order to change efficientnet vision a bit, not to break it
#     optimizer=tf.keras.optimizers.Adam(1e-5),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# #teach 10 more epoches in order to reach higher accuracy
# #first 20 epoches = classification, next 10 epoches = fine tuning
# history_fine = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=30,
#     initial_epoch=history.epoch[-1]
# )

# test_loss, test_acc = model.evaluate(test_ds)
# print("Test accuracy after fine-tuning:", test_acc)

# ---------- visualisation ----------

class_names = train_ds.class_names

# choose one dog and predict its breed 
def show_prediction(model, dataset): # method to show how prediction works
    img_batch, label_batch = next(iter(dataset)) # dataset = (img_batch, label_batch), next takes the first batch

    index = np.random.randint(len(img_batch)) # random dog from the batch
    img = img_batch[index] # random dog image
    label = label_batch[index].numpy() # and its breed label

    img_expanded = np.expand_dims(img, axis=0) # image format
    pred_probs = model.predict(img_expanded, verbose=0)[0] # array of predictions
    pred_class = np.argmax(pred_probs) # choose the highest probability

    true_name = class_names[label] # real dog breed
    pred_name = class_names[pred_class] # predicted breed
    correct = (label == pred_class) # check if it's equal

    plt.figure(figsize=(4,4))
    plt.imshow(img.numpy().astype("uint8"))
    plt.axis("off")
    plt.title(
        f"Prediction: {pred_name}\n"
        f"Actual: {true_name}\n"
        f"Correct: {correct}"
    )
    plt.show()

    return img, label, pred_class # random dog image, its label and guessed label

# if predicted wrong, this method will show a dog from the race predicted incorrectly
def show_example_from_class(class_name, images_root): 
    class_dir = os.path.join(images_root, class_name) # folder address
    files = os.listdir(class_dir) # all images in the folder
    example = np.random.choice(files) # choose one picture from the folder

    img_path = os.path.join(class_dir, example)
    img = image.load_img(img_path)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Example of predicted class: {class_name}")
    plt.show()

# what areas of the image did the model look at when deciding on a breed
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    effnet = model.get_layer("efficientnetb0") # layer which looks at the image parts

    input_tensor = effnet.input

    last_conv_layer = effnet.get_layer(last_conv_layer_name) # last moment when network looks at the image
    last_conv_output = last_conv_layer.output

    x = effnet.output # conv layer output
    for layer in model.layers[2:]:  
        x = layer(x)

    predictions = x # classes predictions

    grad_model = tf.keras.models.Model(
        inputs=input_tensor,
        outputs=[last_conv_output, predictions]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0]) # predicted class
        class_channel = preds[:, pred_index] # probability of this class

    grads = tape.gradient(class_channel, conv_outputs) # gradients = which features had the biggest influence

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1) # important_features * their_importance

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

# apply heatmap on the image
def show_gradcam(img, heatmap, alpha=0.4):
    if isinstance(img, tf.Tensor):
      img = img.numpy()
    img = img.astype("uint8")

    # resize heatmap to the image
    heatmap_resized = tf.image.resize(
        np.expand_dims(heatmap, -1),
        (img.shape[0], img.shape[1])
    ).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    # color heatmap
    cmap = plt.cm.get_cmap('jet') # blue = less important, red = very important, yellow = average importance
    norm = plt.Normalize(vmin=0, vmax=1)
    colored_heatmap = cmap(norm(heatmap_resized))
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)

    # put heatmap on the image
    superimposed = (colored_heatmap * alpha + img * (1 - alpha)).astype("uint8")

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original Image")

    # what was important and what wasn't
    plt.subplot(1,3,2)
    im = plt.imshow(heatmap_resized, cmap='jet') 
    plt.axis("off")
    plt.title("Heatmap")
    plt.colorbar(im, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)

    # where the model was actually looking
    plt.subplot(1,3,3)
    plt.imshow(superimposed)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.show()

# ---------- show 10 predictions ----------
model(tf.zeros((1, 224, 224, 3)))
for i in range(10): # 10 random dogs from test dataset
  img, true_label, pred_class = show_prediction(
      model, test_ds, images_root=images_path
  )

  if pred_class != true_label:
      show_example_from_class(class_names[pred_class], images_path)

  img_expanded = np.expand_dims(img, axis=0)

  heatmap = make_gradcam_heatmap(img_expanded, model, "top_conv")

  show_gradcam(img, heatmap)

# ---------- plot 1: total number of attempts and number of correct guesses on the test set plot 1: total number of attempts and number of correct guesses on the test set ----------

y_true = [] # actual breeds
y_pred = [] # predicted breeds

for images, labels in test_ds:
    preds = model.predict(images, verbose=0) # array of predictions
    pred_classes = np.argmax(preds, axis=1) # predicted breeds
    y_true.extend(labels.numpy()) 
    y_pred.extend(pred_classes) 

y_true = np.array(y_true)
y_pred = np.array(y_pred)

total_attempts = len(y_true) # total correct attempts
correct_predictions = np.sum(y_true == y_pred)
incorrect_predictions = total_attempts - correct_predictions

plt.figure(figsize=(6, 4))
plt.bar(['Correct', 'Incorrect'], [correct_predictions, incorrect_predictions], color=['green', 'red'])
plt.title(f'Total number of attempts: {total_attempts}, correct guesses: {correct_predictions}')
plt.ylabel('Quantity')
plt.show()

# ---------- plot 2: three races with the most errors ----------

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(pred_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

errors = y_true != y_pred
error_counts = Counter(y_true[errors]) # y_true[errors] array of wrong predictions, Counter counts errors for each breed

# most_common(3) returns three breeds with most errors, then turn in into breed labels
top3_error_classes = [class_names[idx] for idx, _ in error_counts.most_common(3)]

plt.figure(figsize=(12, 4))
for i, breed in enumerate(top3_error_classes):
    breed_idx = [idx for idx, label in enumerate(y_true) if class_names[label] == breed]
    correct = np.sum(y_pred[breed_idx] == y_true[breed_idx])
    total = len(breed_idx)
    incorrect = total - correct

    plt.subplot(1, 3, i+1)
    plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
    plt.title(f'{breed}\nCorrect: {correct}/{total}')
    plt.ylabel('Quantity')

plt.tight_layout()
plt.show()

# ---------- plot 3: what other breeds does the model most often confuse with the most incorrect breed ----------

worst_breed_idx = top3_error_classes[0]  
worst_idx = [idx for idx, label in enumerate(y_true) if class_names[label] == worst_breed_idx] # all images of the worst guessed dog in the test dataset
wrong_preds = y_pred[worst_idx][y_pred[worst_idx] != y_true[worst_idx]] # all breeds which was confused with the worst guessed dog, we filter only breeds where network have made a mistake


confusion_counts = Counter([class_names[p] for p in wrong_preds]) # how many times does each breed appear in errors

plt.figure(figsize=(8,4))
plt.bar(confusion_counts.keys(), confusion_counts.values(), color='orange')
plt.xticks(rotation=45, ha='right')
plt.title(f'What breeds are they confused with? {worst_breed_idx}')
plt.ylabel('Number of mistakes')
plt.show()

# ---------- Guess my dog ----------

def predict_external_image(img_path, model, class_names, img_size=(224, 224), top_k=5):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) # turn image into (224,224,3)
    img_array = np.expand_dims(img_array, axis=0) # (1,224,224,3) for the model
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    predictions = model.predict(img_array)[0] # array of predictions
    top_indices = predictions.argsort()[-top_k:][::-1] # 5 most possible predictions

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("External image")
    plt.show()

    print("Top predictions:")
    for i in top_indices: # 5 most possible predictions and it's probability percentage
        print(f"{class_names[i]} — {predictions[i]*100:.2f}%")

    return img, img_array, predictions

img_path = "/content/dog5.jpg"

img, img_array, preds = predict_external_image(
    img_path=img_path,
    model=model,
    class_names=class_names,
    top_k=5
)

heatmap = make_gradcam_heatmap(img_array, model, "top_conv")

show_gradcam(img_array[0], heatmap)