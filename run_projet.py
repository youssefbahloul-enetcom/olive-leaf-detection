"""
Détection en temps réel des maladies des feuilles d'olivier
Usage:
    python run_projet.py              # Utilise le modèle 2 par défaut
    python run_projet.py --model 1    # Modèle CNN basique
    python run_projet.py --model 2    # Modèle MobileNetV2 (recommandé)
    python run_projet.py --camera 1   # Utilise une caméra externe
Touches:
    q  →  Quitter
    s  →  Capturer une screenshot
    m  →  Basculer entre les modèles
    p  →  Pause/Reprendre la détection
"""

import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime
from tensorflow.keras.models import load_model

# ============================================================
# Arguments CLI
# ============================================================
parser = argparse.ArgumentParser(description="Détection en temps réel - Feuille d'olivier")
parser.add_argument("--model", type=int, choices=[1, 2], default=2,
                    help="Numéro du modèle à utiliser (1=CNN basique, 2=MobileNetV2)")
parser.add_argument("--camera", type=int, default=0,
                    help="Index de la caméra (0=webcam intégrée, 1=caméra externe)")
parser.add_argument("--threshold-high", type=float, default=0.75,
                    help="Seuil supérieur pour 'Malade' (défaut: 0.75)")
parser.add_argument("--threshold-low", type=float, default=0.25,
                    help="Seuil inférieur pour 'Saine' (défaut: 0.25)")
args = parser.parse_args()

# ============================================================
# Constantes
# ============================================================
MODEL_PATHS = {
    1: "peacock_model1.h5",
    2: "peacock_model2.h5"
}
MODEL_NAMES = {
    1: "CNN Basique (~85%)",
    2: "MobileNetV2 (~95%)"
}
IMG_SIZE = 224
PREDICT_EVERY_N_FRAMES = 8   # Prédit 1 frame sur N pour alléger le CPU
SCREENSHOTS_DIR = "screenshots"

# Couleurs BGR
COLOR_GREEN  = (50, 220, 50)
COLOR_RED    = (50, 50, 220)
COLOR_YELLOW = (30, 200, 230)
COLOR_WHITE  = (240, 240, 240)
COLOR_BLACK  = (10, 10, 10)
COLOR_DARK   = (30, 30, 30)
COLOR_CYAN   = (200, 200, 50)

# ============================================================
# Chargement du modèle
# ============================================================
def load_selected_model(model_num):
    path = MODEL_PATHS[model_num]
    if not os.path.exists(path):
        print(f"❌ Modèle '{path}' introuvable. Assurez-vous d'être dans le bon répertoire.")
        exit(1)
    print(f"✅ Chargement du modèle {model_num} ({MODEL_NAMES[model_num]})...")
    m = load_model(path)
    print(f"✅ Modèle chargé avec succès.")
    return m

current_model_num = args.model
model = load_selected_model(current_model_num)

# ============================================================
# Prédiction
# ============================================================
def predict_leaf(image, mdl, threshold_high, threshold_low):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = mdl.predict(img, verbose=0)
    confidence = float(pred[0][0])
    if confidence > threshold_high:
        label = "Malade : Oeil de Paon"
        color = COLOR_RED
        status = "INFECTED"
    elif confidence < threshold_low:
        label = "Feuille Saine"
        color = COLOR_GREEN
        status = "HEALTHY"
    else:
        label = "Incertain"
        color = COLOR_YELLOW
        status = "UNCERTAIN"
    return label, confidence, color, status

# ============================================================
# Rendu UI
# ============================================================
def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.5, radius=10):
    """Dessine un rectangle semi-transparent avec coins arrondis."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def draw_confidence_bar(img, x, y, width, height, confidence, label_color):
    """Dessine une barre de confiance."""
    # Fond
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Remplissage
    fill_w = int(width * confidence)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), label_color, -1)
    # Bordure
    cv2.rectangle(img, (x, y), (x + width, y + height), COLOR_WHITE, 1)
    # Texte pourcentage
    pct_text = f"{confidence*100:.1f}%"
    cv2.putText(img, pct_text, (x + width + 8, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

def draw_guide_box(img, h, w):
    """Dessine un cadre de guidage central."""
    box_w, box_h = int(w * 0.5), int(h * 0.5)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    corner = 20
    thickness = 2
    clr = (180, 180, 180)
    # Coins du rectangle guide
    cv2.line(img, (x1, y1), (x1 + corner, y1), clr, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner), clr, thickness)
    cv2.line(img, (x2, y1), (x2 - corner, y1), clr, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner), clr, thickness)
    cv2.line(img, (x1, y2), (x1 + corner, y2), clr, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner), clr, thickness)
    cv2.line(img, (x2, y2), (x2 - corner, y2), clr, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner), clr, thickness)

def draw_hud(frame, label, confidence, color, status, fps, model_num, paused):
    """Dessine tout l'interface HUD sur la frame."""
    h, w = frame.shape[:2]

    # ── Bandeau supérieur ──────────────────────────────────────
    draw_rounded_rect(frame, 0, 0, w, 70, COLOR_DARK, alpha=0.7)

    # Titre
    cv2.putText(frame, "OLIVE LEAF DETECTOR", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2)

    # Modèle actif
    model_text = f"Model {model_num}: {MODEL_NAMES[model_num]}"
    cv2.putText(frame, model_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # FPS (droite)
    fps_text = f"FPS: {fps:.1f}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, fps_text, (w - fps_size[0] - 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

    # PAUSE indicator
    if paused:
        cv2.putText(frame, "⏸ PAUSE", (w - 100, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1)

    # ── Cadre de guidage central ───────────────────────────────
    draw_guide_box(frame, h, w)
    cv2.putText(frame, "Placez la feuille ici", ((w - 170) // 2, h // 2 + int(h * 0.27)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ── Bandeau de résultat (bas) ──────────────────────────────
    result_y = h - 100
    draw_rounded_rect(frame, 0, result_y, w, h, COLOR_DARK, alpha=0.75)

    if label and not paused:
        # Label principal
        cv2.putText(frame, label, (12, result_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Barre de confiance
        bar_x, bar_y = 12, result_y + 45
        bar_w = w - 140
        draw_confidence_bar(frame, bar_x, bar_y, bar_w, 20, confidence, color)

        # Texte confiance détaillé
        cv2.putText(frame, f"Confiance: {confidence:.3f}", (12, result_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    elif paused:
        cv2.putText(frame, "Détection en pause", (12, result_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)

    # ── Touches clavier (bas droite) ─────────────────────────
    help_texts = ["q:Quitter", "s:Screenshot", "m:Modele", "p:Pause"]
    for i, t in enumerate(help_texts):
        cv2.putText(frame, t, (w - 120, result_y + 18 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

# ============================================================
# Création dossier screenshots
# ============================================================
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ============================================================
# Ouverture de la caméra
# ============================================================
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"❌ Impossible d'ouvrir la caméra {args.camera}.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Caméra ouverte. Appuyez sur 'q' pour quitter.")
print("   Touches: s=screenshot | m=changer modèle | p=pause")

# ============================================================
# Variables d'état
# ============================================================
frame_count    = 0
fps            = 0.0
fps_timer      = time.time()
fps_frame_cnt  = 0

last_label      = None
last_confidence = 0.0
last_color      = COLOR_WHITE

paused         = False

# ============================================================
# Boucle principale
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame non reçue, nouvelle tentative...")
        time.sleep(0.05)
        continue

    frame_count += 1

    # ── Calcul FPS ──────────────────────────────────────────
    fps_frame_cnt += 1
    elapsed = time.time() - fps_timer
    if elapsed >= 1.0:
        fps = fps_frame_cnt / elapsed
        fps_frame_cnt = 0
        fps_timer = time.time()

    # ── Prédiction (throttlée) ───────────────────────────────
    if not paused and frame_count % PREDICT_EVERY_N_FRAMES == 0:
        try:
            last_label, last_confidence, last_color, _ = predict_leaf(
                frame, model, args.threshold_high, args.threshold_low
            )
        except Exception as e:
            print(f"⚠️ Erreur prédiction: {e}")

    # ── Dessin HUD ───────────────────────────────────────────
    draw_hud(frame, last_label, last_confidence, last_color,
             None, fps, current_model_num, paused)

    cv2.imshow("🌿 Olive Leaf Disease Detector", frame)

    # ── Touches ─────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("👋 Fermeture...")
        break

    elif key == ord('s'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SCREENSHOTS_DIR, f"capture_{ts}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot sauvegardée: {filename}")

    elif key == ord('m'):
        # Basculer entre les modèles
        current_model_num = 2 if current_model_num == 1 else 1
        print(f"🔄 Changement vers modèle {current_model_num}...")
        model = load_selected_model(current_model_num)
        last_label = None

    elif key == ord('p'):
        paused = not paused
        status_str = "en pause" if paused else "reprise"
        print(f"⏸ Détection {status_str}")

# ============================================================
# Libération
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("✅ Application fermée proprement.")
