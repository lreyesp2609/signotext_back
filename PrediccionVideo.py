import os
import tensorflow as tf
import numpy as np
import base64
import cv2
import tempfile

def predecir_video_base64(nombre_archivo_modelo, video_base64):
    # Verificar que el modelo existe
    model_path = f"./ModelosPalabras/{nombre_archivo_modelo}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")
    
    # Cargar el modelo
    modelo = tf.keras.models.load_model(model_path)
    
    # Decodificar el video base64
    video_decodificado = base64.b64decode(video_base64)
    
    # Guardar temporalmente el video en un archivo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_video:
        temp_video.write(video_decodificado)
        video_path = temp_video.name
    
    # Leer el video usando OpenCV
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensionar el frame al tamaño esperado por el modelo
        frame = cv2.resize(frame, (150, 150))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        frames.append(frame)

    cap.release()
    
    # Convertir los frames en un arreglo numpy y normalizarlos
    frames = np.array(frames) / 255.0  # Normalizar los valores del pixel entre 0 y 1
    frames = np.array(frames)  # Ya tienes la dimensión de lote implícita

    # Realizar la predicción en el video (sobre los frames)
    prediccion = modelo.predict(frames)
    
    # Limpiar archivo temporal
    os.remove(video_path)
    
    print("Predicción realizada:", prediccion)
    return prediccion
