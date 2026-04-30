import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================================
# Charger le modèle
# ================================
model = load_model("peacock_model2.h5")

# ================================
# Fonction de prédiction
# ================================
def predict_leaf(image, model):
    # Prétraitement
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prédiction
    pred = model.predict(img, verbose=0)
    confidence = pred[0][0]

    # ================================
    # Seuil de confiance
    # ================================
    if confidence > 0.75:
        return "Malade : Oeil de paon", confidence
    elif confidence < 0.25:
        return "Saine", confidence
    else:
        return None, confidence  # incertain → rien afficher


# ================================
# Ouvrir la caméra
# ================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Caméra non détectée")
    exit()

# ================================
# Boucle vidéo
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prédiction
    result, conf = predict_leaf(frame, model)

    # ================================
    # Affichage conditionnel
    # ================================
    if result is not None:
        cv2.putText(
            frame,
            f"{result} ({conf:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    else:
        # Optionnel : afficher message si rien détecté
        cv2.putText(
            frame,
            "Aucune feuille detectee",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Afficher la fenêtre
    cv2.imshow("Detection feuille d'olivier", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================
# Libération
# ================================
cap.release()
cv2.destroyAllWindows()