def entrenar_modelo(nombre_archivo):
    import os
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import json
    import numpy as np
    #import tensorflowjs as tfjs
    # Definir la ruta de las carpetas de entrenamiento y prueba
    train_data_dir = './Senias'
    validation_data_dir = './Senias'

    # Contar el número de carpetas/clases en el directorio de datos
    num_classes = len(os.listdir(train_data_dir))

    # Definir parámetros de preprocesamiento y generador de datos
    img_height, img_width = 150, 150
    batch_size = 32
    epochs = 10

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

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
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Guardar el modelo entrenado
    model.save(f"./Modelos/{nombre_archivo}.keras")