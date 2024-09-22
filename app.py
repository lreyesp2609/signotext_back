import psycopg2
from jsonrpc import JSONRPCResponseManager, dispatcher
from werkzeug.wrappers import Request, Response
import json
from flask import Flask
from flask_cors import CORS
import numpy as np
#imports mapReduce
from multiprocessing import Pool
from collections import defaultdict
import os
import glob
import random

from EntrenamientoVideo import entrenar_modelo_videos
from PrediccionVideo import predecir_video_base64
app = Flask(__name__)
CORS(app)
import base64
import os
from datetime import datetime
from Entrenamiento import *
from Prediccion import predecir_imagen_base64
import shutil

@dispatcher.add_method
def recibirJsonSenias(nombreSenia, imagenesSenia):
    nombre_carpeta = "./Senias/"+nombreSenia
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
    for imagen in imagenesSenia:
        # Obtener el contenido base64 de la imagen
        contenido_base64 = imagen["img"].split(",")[1]
        # Decodificar el contenido base64
        contenido_bytes = base64.b64decode(contenido_base64)
        fecha_actual = datetime.now().strftime("%Y%m%d%H%M%S") #para evitar que se puedan repetir
        with open(os.path.join(nombre_carpeta, f"{imagen['id']}_{fecha_actual}.png"), "wb") as archivo:
            archivo.write(contenido_bytes)

@dispatcher.add_method
def recibirJsonSeniasPalabras(nombreSenia, videosSenia):
    nombre_carpeta = f"./SeniasPalabras/{nombreSenia}"
    try:
        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)
    except Exception as e:
        print(f"Error al crear la carpeta: {e}")
        return  # Salir de la función si hay un error

    for video in videosSenia:
        # Obtener el contenido base64 del video
        contenido_base64 = video["video"].split(",")[1]
        # Decodificar el contenido base64
        contenido_bytes = base64.b64decode(contenido_base64)
        fecha_actual = datetime.now().strftime("%Y%m%d%H%M%S")  # para evitar repeticiones
        with open(os.path.join(nombre_carpeta, f"{video['id']}_{fecha_actual}.webm"), "wb") as archivo:
            archivo.write(contenido_bytes)

#Funcion vacia para darle la funcionalidad de generar el modelo 
@dispatcher.add_method
def GenerarModelo():
    #EN LA BD GUARDAR TAMBIEN EL NOMBRE DEL MODELO PARA LUEGO ADMINISTRARLO
    print("AQUI GENERAR EL MODELO Y GUARDARLO EN UNA CARPETA DONDE SE ENCUENTREN TODOS LOS MODELOS CON EL NOMBRE POR FECHA")
    #La carpeta principal puede ser: ./Modelo/ y dentro tendrian que ir los modelos generados con la fecha como nombre
    fecha_actual = datetime.now().strftime("%Y%m%d%H%M%S") #para evitar que se puedan repetir
    entrenar_modelo("Modelo"+fecha_actual)

@dispatcher.add_method
def GenerarModeloPalabra():
    # Guardar el modelo en una carpeta con la fecha actual en el nombre
    print("Generando el modelo y guardándolo en una carpeta por fecha...")
    
    # Crear carpeta con nombre según la fecha actual
    fecha_actual = datetime.now().strftime("%Y%m%d%H%M%S")

    # Entrenar el modelo (reutilizando la función `entrenar_modelo`)
    entrenar_modelo_videos(f"Modelo_{fecha_actual}")
    
@dispatcher.add_method
def PredecirImagen(imagen_base64, ModeloAPredecir):
    # Primero crear un diccionario con las carpetas para predecir el modelo
    ruta = './Senias/'
    diccionario_clases = [nombre for nombre in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, nombre))]
    print(diccionario_clases)

    def add_padding(s):
        return s + '=' * ((4 - len(s) % 4) % 4)

    # Aplicar el padding a la imagen base64
    imagen_base64_padded = add_padding(imagen_base64)

    print(f"Modelo: {ModeloAPredecir}")
    print(f"Longitud de la imagen base64: {len(imagen_base64_padded)}")
    
    try:
        prediccion = predecir_imagen_base64(ModeloAPredecir, imagen_base64_padded)
        
        print(f"Predicción: {prediccion}")
        
        # Obtener el índice del valor máximo en el array
        resultado_prediccion = np.array(prediccion)
        indice_max_probabilidad = np.argmax(resultado_prediccion)

        # Obtener la clase predicha
        clase_predicha = diccionario_clases[indice_max_probabilidad]

        print(f"Clase predicha: {clase_predicha}")
        return clase_predicha
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return str(e)
    
@dispatcher.add_method
def PredecirVideo(video_base64, ModeloAPredecir):
    # Crear un diccionario con las carpetas para predecir el modelo
    ruta = './SeniasPalabras/'
    diccionario_clases = [nombre for nombre in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, nombre))]
    print(f"Clases disponibles: {diccionario_clases}")

    # Aplicar el padding a la cadena base64 si es necesario
    def add_padding(s):
        return s + '=' * ((4 - len(s) % 4) % 4)

    # Aplicar el padding al video base64
    video_base64_padded = add_padding(video_base64)

    print(f"Modelo: {ModeloAPredecir}")
    print(f"Longitud del video base64: {len(video_base64_padded)}")
    
    try:
        prediccion = predecir_video_base64(ModeloAPredecir, video_base64_padded)
        
        print(f"Predicción: {prediccion}")
        
        # Obtener el promedio de las predicciones
        resultado_prediccion = np.array(prediccion)
        promedio_predicciones = np.mean(resultado_prediccion, axis=0)  # Promedio por clase
        
        # Obtener la clase predicha con la mayor probabilidad por frame
        indices_max_probabilidad = np.argmax(resultado_prediccion, axis=1)  # Índices de la clase más alta para cada frame
        clases_predichas = [diccionario_clases[indice] for indice in indices_max_probabilidad]
        # Realizar una votación
        clase_predicha = max(set(clases_predichas), key=clases_predichas.count)  # Clase más común
        
        print(f"Clase predicha: {clase_predicha}")
        return clase_predicha  # Devolver solo la clase predicha
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return str(e)

#funcion para listar en un JSON todas las carpetas que esten en Senias con sus nombres 
@dispatcher.add_method
def listar_carpetas():
    directorio = './Senias/'
    carpetas = [{'nombre': nombre} for nombre in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, nombre))]
    return json.dumps(carpetas)

#funcion para listar las imagenes de un directorio para luego poder eliminarlas skere modo diablo
@dispatcher.add_method
def listar_imagenes_en_carpeta(NombreSenia):
    imagenes = []
    ruta_carpeta = os.path.join("./Senias/", NombreSenia)
    for nombre_imagen in os.listdir(ruta_carpeta):
        if nombre_imagen.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            with open(os.path.join(ruta_carpeta, nombre_imagen), "rb") as imagen_archivo:
                base64_image = base64.b64encode(imagen_archivo.read()).decode('utf-8')
                imagenes.append({"nombreImagen": nombre_imagen, "base64": base64_image})
    return json.dumps(imagenes)

#Funcion para eliminar una imagen skere modo diablo
@dispatcher.add_method
def eliminar_imagen(nombre_imagen, carpeta):
    ruta_imagen = os.path.join("./Senias/", carpeta, nombre_imagen)
    if os.path.exists(ruta_imagen):
        os.remove(ruta_imagen)
        return f"Imagen {nombre_imagen} eliminada correctamente."
    else:
        return f"La imagen {nombre_imagen} no existe en la carpeta {carpeta}."

#Funcion para renombrar una carpeta o una senia para el modelo
@dispatcher.add_method
def renombrar_carpeta(nombre_actual, nuevo_nombre):
    ruta_actual = os.path.join("./Senias/", nombre_actual)
    ruta_nueva = os.path.join("./Senias/", nuevo_nombre)
    if os.path.exists(ruta_actual):
        os.rename(ruta_actual, ruta_nueva)
        return f"Carpeta {nombre_actual} renombrada a {nuevo_nombre}."
    else:
        return f"La carpeta {nombre_actual} no existe."
  
#Funcion para eliminar una senia de manera definitiva con todo y su contenido 
@dispatcher.add_method
def eliminar_carpeta(nombre_carpeta):
    ruta_carpeta = os.path.join("./Senias/", nombre_carpeta)
    if os.path.exists(ruta_carpeta):
        shutil.rmtree(ruta_carpeta)
        return f"Carpeta {nombre_carpeta} y su contenido eliminados correctamente."
    else:
        return f"La carpeta {nombre_carpeta} no existe."

#funcion para listar los modelos para usarlos en el test o eliminarlos skere 
@dispatcher.add_method
def listar_modelos():
    ruta_carpeta = "./Modelos/"
    modelos = []
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith('.keras'):
            modelos.append({"nombre": nombre_archivo})
    return json.dumps(modelos)

@dispatcher.add_method
def listar_modelos_videos():
    ruta_carpeta = "./ModelosPalabras/"
    modelos = []
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith('.keras'):
            modelos.append({"nombre": nombre_archivo})
    return json.dumps(modelos)

@Request.application
def application(request):
    response = JSONRPCResponseManager.handle(
        request.get_data(as_text=True),
        dispatcher
    )
    #Aun no tiene la funcion de los tokens y creo que es mejor de momento skere
    # Configuración de los encabezados CORS
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    return Response(response.json, content_type='application/json', headers=headers)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 4000, application)