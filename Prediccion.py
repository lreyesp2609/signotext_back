import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64

def predecir_imagen_base64(nombre_archivo_modelo, imagen_base64):
    # Cargar el modelo
    model_path = f"./Modelos/{nombre_archivo_modelo}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo {model_path}")
    modelo = tf.keras.models.load_model(model_path)

    # Decodificar la imagen base64
    imagen_decodificada = base64.b64decode(imagen_base64)
    imagen = tf.image.decode_image(imagen_decodificada, channels=3)
    imagen = tf.image.resize(imagen, [150, 150])  # Ajustar al tamaño de entrada del modelo
    imagen = img_to_array(imagen)
    imagen = np.expand_dims(imagen, axis=0)

    # Realizar la predicción
    prediccion = modelo.predict(imagen)
    print(prediccion)
    return prediccion
