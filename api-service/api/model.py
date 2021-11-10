import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_hub as hub
import albumentations as A
from functools import partial


AUTOTUNE = tf.data.experimental.AUTOTUNE
local_experiments_path = "/persistent/experiments"
best_model = None
best_model_id = None
prediction_model = None
data_details = None
image_width = 224
image_height = 224
num_channels = 3

artist_label = ['Albrecht_Durer', 'Alfred_Sisley', 'Amedeo_Modigliani', 'Andrei_Rublev', 'Andy_Warhol',
                'Camille_Pissarro', 'Caravaggio', 'Claude_Monet', 'Diego_Rivera', 'Diego_Velazquez',
                'Edgar_Degas', 'Edouard_Manet', 'Edvard_Munch', 'El_Greco', 'Eugene_Delacroix', 'Francisco_Goya',
                'Frida_Kahlo', 'Georges_Seurat', 'Giotto_di_Bondone', 'Gustave_Courbet', 'Gustav_Klimt',
                'Henri_de_Toulouse-Lautrec', 'Henri_Matisse', 'Henri_Rousseau', 'Hieronymus_Bosch', 'Jackson_Pollock',
                'Jan_van_Eyck', 'Joan_Miro', 'Kazimir_Malevich', 'Leonardo_da_Vinci', 'Marc_Chagall', 'Michelangelo',
                'Mikhail_Vrubel', 'Pablo_Picasso', 'Paul_Cezanne', 'Paul_Gauguin', 'Paul_Klee', 'Peter_Paul_Rubens',
                'Pierre-Auguste_Renoir', 'Pieter_Bruegel', 'Piet_Mondrian', 'Raphael', 'Rembrandt', 'Rene_Magritte',
                'Salvador_Dali', 'Sandro_Botticelli', 'Titian', 'Vasiliy_Kandinskiy', 'Vincent_van_Gogh', 'William_Turner']

project_path = 'C:/Users/Lei/Desktop/Fall2021/AP215_advanced_practical_data_science/exercise/exercise6/artist-application'


def load_prediction_model():
    print("Loading Model...")
    global prediction_model

    best_model_path = os.path.join(
        "/persistent", "model_mobilenetv2_train_base_True.hdf5")

    print("best_model_path:", best_model_path,)

    prediction_model = tf.keras.models.load_model(
        best_model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    print(prediction_model.summary())

    # data_details_path = os.path.join(
    #     local_experiments_path, best_model["user"], best_model["experiment"], "data_details.json")

    # # Load data details
    # with open(data_details_path, 'r') as json_file:
    #     data_details = json.load(json_file)


# def check_model_change():
#     global best_model, best_model_id
#     best_model_json = os.path.join(local_experiments_path, "best_model.json")
#     if os.path.exists(best_model_json):
#         with open(best_model_json) as json_file:
#             best_model = json.load(json_file)

#         if best_model_id != best_model["experiment"]:
#             load_prediction_model()
#             best_model_id = best_model["experiment"]


def load_preprocess_image_from_path(image_path):
    print("Image", image_path)

    image_width = 224
    image_height = 224
    num_channels = 3

    # Prepare the data

    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        # image = tf.image.resize(image, [image_height, image_width])
        return image

    # Normalize pixels
    def normalize(image):
        image = image / 255
        return image

    # Image agmentation
    augmentor = A.Compose([
        A.augmentations.geometric.resize.SmallestMaxSize(
            max_size=image_width, p=1),
        A.augmentations.crops.transforms.CenterCrop(image_width, image_height, p=1)]
    )

    def aug(image):
        aug_img = augmentor(image=image)['image']
        aug_img = tf.image.convert_image_dtype(aug_img, 'float32')
        return aug_img

    def center_crop(image):
        aug_img = tf.numpy_function(func=aug, inp=[image], Tout=tf.float32)
        return aug_img

    test_data = tf.data.Dataset.from_tensor_slices(([image_path]))
    test_data = test_data.map(load_image, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(center_crop, num_parallel_calls=AUTOTUNE)
    test_data = test_data.repeat(1).batch(1)

    return test_data


def make_prediction(image_path):
    load_prediction_model()

    # Load & preprocess
    test_data = load_preprocess_image_from_path(image_path)

    # Make prediction
    prediction = prediction_model.predict(test_data)
    idx = prediction.argmax(axis=1)[0]
    prediction_label = artist_label[idx]

    if prediction_model.layers[-1].activation.__name__ != 'softmax':
        prediction = tf.nn.softmax(prediction).numpy()
        print(prediction)

    # poisonous = False
    # if prediction_label == "amanita":
    #     poisonous = True

    return {
        "input_image_shape": str(test_data.element_spec.shape),
        "prediction_shape": prediction.shape,
        "prediction_label": prediction_label,
        "prediction": prediction.tolist(),
        "accuracy": round(np.max(prediction)*100, 2)
        # "poisonous": poisonous
    }
