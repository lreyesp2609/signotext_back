import os
import tensorflow as tf
import cv2
import numpy as np
import re
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

# Función para limpiar nombres de carpetas con caracteres especiales
def limpiar_nombre(nombre):
    return re.sub(r'[<>:"/\\|?*]', '_', nombre)  # Reemplazar caracteres inválidos con '_'

# Función para cargar el mapeo entre nombres originales y sanitizados
def cargar_mapeo_nombres():
    archivo_mapeo = './SeniasPalabras/mapeo_nombres.json'
    if os.path.exists(archivo_mapeo):
        with open(archivo_mapeo, 'r') as archivo:
            return json.load(archivo)
    return {}

# Función para guardar mapeo de nombres
def guardar_mapeo_nombre(nombre_original, nombre_limpio):
    archivo_mapeo = './SeniasPalabras/mapeo_nombres.json'
    if os.path.exists(archivo_mapeo):
        with open(archivo_mapeo, 'r') as archivo:
            mapeo = json.load(archivo)
    else:
        mapeo = {}

    mapeo[nombre_limpio] = nombre_original
    with open(archivo_mapeo, 'w') as archivo:
        json.dump(mapeo, archivo)

# Función para extraer frames de un video
def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_frames

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (150, 150))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return np.array(frames)

# Función para cargar los datos desde las carpetas de clases
def load_data(data_dir):
    X = []
    y = []
    mapeo_nombres = cargar_mapeo_nombres()
    
    # Listar solo carpetas, ignorando archivos como mapeo_nombres.json
    class_names = [nombre for nombre in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, nombre))]
    
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        nombre_original = mapeo_nombres.get(class_name, class_name)  # Recuperar nombre original si está mapeado
        for video_file in os.listdir(class_dir):
            if video_file.endswith('.webm'):
                video_path = os.path.join(class_dir, video_file)
                frames = extract_frames(video_path)
                X.extend(frames)
                y.extend([class_index] * len(frames))
    
    return np.array(X), np.array(y), class_names

# Función para entrenar el modelo con los videos
def entrenar_modelo_videos(nombre_archivo):
    # Definir la ruta de las carpetas de entrenamiento
    train_data_dir = './SeniasPalabras'

    # Cargar datos
    X, y, class_names = load_data(train_data_dir)
    num_classes = len(class_names)

    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocesar los datos
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)

    # Convertir las etiquetas a one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)

    # Definir parámetros de entrenamiento
    img_height, img_width = 150, 150
    batch_size = 32
    epochs = 10

    # Definir modelo de red neuronal convolucional
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

    # Guardar el modelo entrenado
    if not os.path.exists('./ModelosPalabras'):
        os.makedirs('./ModelosPalabras')

    model.save(f"./ModelosPalabras/{nombre_archivo}.keras")
    print(f"Modelo guardado como {nombre_archivo}.keras")
