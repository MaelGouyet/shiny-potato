import cv2
import mediapipe as mp
import numpy as np

# Charger l'image traitée par rembg (avec fond transparent)
image_path = "visage_sans_bg.png"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Charger l'image avec transparence
h, w, _ = image.shape

# Initialiser Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Convertir en RGB pour Mediapipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    # Récupérer les points du visage
    for face_landmarks in results.multi_face_landmarks:
        # Liste des coordonnées des points du visage
        points = []
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))

        # Sélectionner uniquement les points formant le contour du visage
        face_contour = np.array([
            points[10], points[338], points[297], points[332], points[284], points[251],  # Contour gauche
            points[389], points[356], points[454],  # Menton
            points[323], points[361], points[288], points[397], points[365], points[379], points[378],  # Contour droit
            points[400], points[377], points[152]  # Haut du visage
        ], dtype=np.int32)

        # Créer un masque noir
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_contour], 255)  # Remplir le masque avec la forme du visage

        # Appliquer le masque à l'image
        face_only = cv2.bitwise_and(image, image, mask=mask)

        # Sauvegarde du visage avec découpe précise
        cv2.imwrite("face_exact_cut.png", face_only)

print("Visage découpé avec succès !")
