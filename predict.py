import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import cv2

model = load_model("mobilenet_model_1.h5")

train_dir = "archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.15,
    rotation_range=10,       
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

class_names = list(train_gen.class_indices.keys())
print("Classes:", class_names)


def predict_leaf(img_path):
    IMG_SIZE = (128, 128)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    confidence = np.max(preds)

    print("\nPrediction:", class_names[predicted_index])
    print("Confidence:", round(confidence * 100, 2), "%")

    return class_names[predicted_index], confidence

predict_leaf("archive/test/test/TomatoHealthy4.JPG")
