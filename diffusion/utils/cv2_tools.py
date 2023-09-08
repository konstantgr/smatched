import cv2
import numpy as np
from PIL import Image


def extend_face_bbox(img: np.ndarray, bbox: tuple[float, float, float, float]):
    x, y, w, h = bbox
    padding = 100
    # Add padding around the detected face
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    x = max(.0, x)
    y = max(.0, y)
    w = min(img.shape[1] - x, w)
    h = min(img.shape[0] - y, h)
    cropped_face = img[y:y + h, x:x + w]
    resized_face = cv2.resize(cropped_face, (512, 512))
    pil_resized_face = Image.fromarray(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))
    return pil_resized_face


def detect_faces(img: np.ndarray) -> list:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        img, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30)
    )
    return faces
