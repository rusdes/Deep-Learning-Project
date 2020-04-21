import argparse
import json
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image 

image_size = 224

parser = argparse.ArgumentParser(description='Predict Flower Class')

def process_image(image):
    im = tf.cast(image, tf.float32)
    im = tf.image.resize(image, (image_size, image_size))
    im /= 255
    return im.numpy()

def pred(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    y = np.expand_dims(processed_image, axis=0)
    pred = model.predict(y).squeeze()
    pred2 = pred
    arr = np.argpartition(pred, -top_k)[-top_k:]
    arr = np.flip(arr, 0)
    classes = arr
    probs = []
    for i in classes:
        probs.append(pred2[i])
        
    classes += 1
    return probs, classes

if __name__ == '__main__':
    parser.add_argument("filepath", type=str, help='Filepath (e.g. test_images/cautleya_spicata.jpg)')
    parser.add_argument("model", type=str, help='Model (e.g my_model.h5)')
    parser.add_argument("--top_K", type=int, help='Number of classes to show', default=1)
    parser.add_argument("--category_names", type=str, help='Path to a JSON file mapping labels to flower names', default='label_map.json')
    arg = parser.parse_args()
    with open(arg.category_names, 'r') as f:
        class_names = json.load(f)

    # load the model
    reloaded_keras_model = tf.keras.models.load_model(arg.model, custom_objects={'KerasLayer': hub.KerasLayer})
    a, b = pred(arg.filepath, reloaded_keras_model, arg.top_K)
    index = 0
    for i in b:
        print("{}: {}".format(class_names[str(i)], a[index]))
        index += 1
