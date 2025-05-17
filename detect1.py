import argparse
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
from picamera2 import Picamera2
from gtts import gTTS
import os
import threading
from queue import Queue

# Initialisation des variables globales pour le calcul des FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
detection_result_list = []
detection_queue = Queue()  # Queue pour accumuler les objets détectés
detection_lock = threading.Lock()  # Verrou pour gérer l'accès concurrent aux résultats
recently_spoken_objects = set()  # Ensemble pour les objets déjà prononcés récemment
refresh_interval = 10  # Intervalle en secondes pour réinitialiser les objets prononcés

# Initialisation de la caméra
try:
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
except Exception as e:
    print(f"Erreur d'initialisation de la caméra : {e}")
    sys.exit(1)

# Fonction pour convertir le texte en audio et le jouer
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp_audio.mp3")
        os.system("mpg321 temp_audio.mp3")  # Pour lire l'audio
        os.remove("temp_audio.mp3")  # Supprimer le fichier temporaire après lecture
    except Exception as e:
        print(f"Erreur lors de la conversion en audio : {e}")

# Fonction pour enregistrer les résultats de détection en audio
def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    global FPS, COUNTER, START_TIME
    # Calculer les FPS
    if COUNTER % 10 == 0:
        FPS = 10 / (time.time() - START_TIME)
        START_TIME = time.time()

    detection_lock.acquire()  # Verrouiller l'accès à la liste de résultats
    detection_result_list.append(result)
    for detection in result.detections:
        detected_object = detection.categories[0].category_name
        if detected_object not in recently_spoken_objects:  # Vérifier si déjà prononcé récemment
            print(f"Objet détecté : {detected_object}")
            detection_queue.put(detected_object)  # Ajouter l'objet détecté dans la queue
            recently_spoken_objects.add(detected_object)  # Marquer comme prononcé
    COUNTER += 1
    detection_lock.release()  # Libérer le verrou

# Fonction pour vider périodiquement les objets prononcés
def refresh_recently_spoken():
    while True:
        time.sleep(refresh_interval)  # Attendre le délai de rafraîchissement
        recently_spoken_objects.clear()  # Vider les objets prononcés
        print("Réinitialisation des objets prononcés.")

# Fonction pour traiter les objets détectés et lire les résultats par ordre
def process_detection_queue():
    while True:
        if not detection_queue.empty():
            detected_objects = []
            while not detection_queue.empty():
                detected_objects.append(detection_queue.get())

            text = " ".join(detected_objects)
            speak(text)  # Lire les objets détectés
        time.sleep(2)  # Attendre avant de traiter à nouveau

# Fonction principale pour exécuter la détection d'objets
def run(model: str, max_results: int, score_threshold: float, camera_id: int, width: int, height: int) -> None:
    try:
        # Initialisation du modèle de détection d'objets
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=score_threshold,
            result_callback=save_result
        )
        detector = vision.ObjectDetector.create_from_options(options)
    except Exception as e:
        print(f"Erreur d'initialisation du modèle : {e}")
        sys.exit(1)

    # Lancer le traitement des détections et le rafraîchissement des objets en arrière-plan
    threading.Thread(target=process_detection_queue, daemon=True).start()
    threading.Thread(target=refresh_recently_spoken, daemon=True).start()

    while True:
        try:
            im = picam2.capture_array()
            image = cv2.resize(im, (640, 480))
            image = cv2.flip(image, -1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Détecter les objets
            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

            # Afficher les FPS sur l'image
            fps_text = f'FPS = {FPS:.1f}'
            cv2.putText(image, fps_text, (24, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

            # Visualiser le résultat si disponible
            detection_lock.acquire()
            if detection_result_list:
                image = visualize(image, detection_result_list[0])
                detection_result_list.clear()
            detection_lock.release()

            cv2.imshow('Object Detection', image)
            if cv2.waitKey(1) == 27:
                break
        except Exception as e:
            print(f"Erreur lors de la capture ou du traitement de l'image : {e}")
            break

    # Libérer les ressources
    try:
        detector.close()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Erreur lors de la libération des ressources : {e}")

# Fonction principale pour l'argument parsing
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', default='efficientdet_lite0.tflite')
    parser.add_argument('--maxResults', help='Max number of detection results.', default=5)
    parser.add_argument('--scoreThreshold', help='The score threshold of detection results.', type=float, default=0.5)
    parser.add_argument('--cameraId', help='Id of camera.', type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', type=int, default=480)
    args = parser.parse_args()

    run(args.model, int(args.maxResults), args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()
