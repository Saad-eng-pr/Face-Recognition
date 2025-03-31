import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# chargement du détecteur de visages et des modèles de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def getFaceEmbedding(image):
    faces = detector(image)
    if len(faces) == 0:
        return None, None  # retourne None pour l'encodage et la forme si aucun visage n'est détecté
    shape = predictor(image, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return face_descriptor, shape  # renvoyer à la fois l'encodage et les points de repère

def compareFaces(encoding1, encoding2, threshold=0.6):
    return distance.euclidean(encoding1, encoding2) < threshold

def drawLandmarks(image, shape):
    # dessiner les points de repère seu l'image donnée
    for i in range(68):  # 68 landmark points
        x, y = shape.part(i).x, shape.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green dots

def resizeImage(image, target_height=300):
    # redimensionner l'image en conservant le rapport hauteur/largeur et la même hauteur.
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))

def areSamePerson(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        return "impossible de lire les images."

    enc1, shape1 = getFaceEmbedding(img1)
    enc2, shape2 = getFaceEmbedding(img2)

    if enc1 is None or enc2 is None:
        return "les visages ne sont pas détectés."

    show_faces = input("pour afficher les images, écrire yes: ")
    if show_faces == "yes":
        # dessiner des repères faciaux sur des images
        drawLandmarks(img1, shape1)
        drawLandmarks(img2, shape2)

        # redimensionner les deux images à la même hauteur avant de les afficher
        target_height = 300
        img1_resized = resizeImage(img1, target_height)
        img2_resized = resizeImage(img2, target_height)

        # afficher les images
        combined_image = np.hstack((img1_resized, img2_resized))
        cv2.imshow("comparaison des visages", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return "même personne" if compareFaces(enc1, enc2) else "personnes différentes"

print(areSamePerson("img1.jpg", "img5.jpg"))