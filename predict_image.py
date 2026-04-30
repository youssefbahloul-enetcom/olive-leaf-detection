"""
Prédiction sur image(s) statiques - Feuille d'Olivier
Usage:
    python predict_image.py --input mon_image.jpg
    python predict_image.py --input test/
    python predict_image.py --input test/ --model 1 --show
"""

import cv2
import numpy as np
import argparse
import os
import csv
from datetime import datetime
from tensorflow.keras.models import load_model

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True)
parser.add_argument("--model",  type=int, choices=[1, 2], default=2)
parser.add_argument("--output", default="results.csv")
parser.add_argument("--show",   action="store_true")
parser.add_argument("--threshold-high", type=float, default=0.75)
parser.add_argument("--threshold-low",  type=float, default=0.25)
args = parser.parse_args()

MODEL_PATHS = {1: "peacock_model1.h5", 2: "peacock_model2.h5"}
IMG_SIZE    = 224
FORMATS     = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

model_path = MODEL_PATHS[args.model]
if not os.path.exists(model_path):
    print(f"Modele '{model_path}' introuvable."); exit(1)

print(f"Chargement modele {args.model}...")
model = load_model(model_path)
print("Modele charge.\n")

def collect_images(path):
    imgs = []
    if os.path.isfile(path):
        if path.lower().endswith(FORMATS):
            imgs.append(path)
    elif os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(FORMATS):
                imgs.append(os.path.join(path, f))
    else:
        print(f"Chemin invalide: {path}"); exit(1)
    return imgs

def predict(image_bgr):
    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.expand_dims(img, axis=0)
    conf = float(model.predict(img, verbose=0)[0][0])
    if conf > args.threshold_high:
        return "Malade : Oeil de Paon", "INFECTED", conf, "#e74c3c"
    elif conf < args.threshold_low:
        return "Feuille Saine", "HEALTHY", conf, "#27ae60"
    return "Incertain", "UNCERTAIN", conf, "#f39c12"

def show_result(img_bgr, label, conf, color, path):
    if not MATPLOTLIB_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(f"{os.path.basename(path)}\n{label}  |  {conf:.2%}",
                 fontsize=13, color=color, fontweight="bold", pad=10)
    fig.patch.set_facecolor("#1a1a2e")
    bar = fig.add_axes([0.1, 0.04, 0.8, 0.03])
    bar.barh(0, conf, color=color, height=1)
    bar.barh(0, 1 - conf, left=conf, color="#444", height=1)
    bar.set_xlim(0, 1); bar.axis("off")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

images = collect_images(args.input)
if not images:
    print("Aucune image trouvee."); exit(1)

print(f"Traitement de {len(images)} image(s)...\n")
print(f"{'Fichier':<40} {'Resultat':<25} {'Conf':>8}  {'Statut'}")
print("-" * 85)

results = []
counts = {"INFECTED": 0, "HEALTHY": 0, "UNCERTAIN": 0}
icons  = {"INFECTED": "[MALADE]", "HEALTHY": "[SAIN]", "UNCERTAIN": "[?]"}

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Impossible de lire: {img_path}"); continue

    label, status, conf, color = predict(img)
    counts[status] += 1
    fname = os.path.basename(img_path)
    print(f"{fname:<40} {icons[status]} {label:<22} {conf:>7.2%}  {status}")

    results.append({
        "file": img_path, "filename": fname,
        "label": label, "status": status,
        "confidence": round(conf, 4),
        "model": args.model,
        "timestamp": datetime.now().isoformat()
    })

    if args.show:
        show_result(img, label, conf, color, img_path)

total = len(results)
print("\n" + "=" * 85)
print(f"RESUME — {total} image(s) analysee(s)")
print(f"  Saines    : {counts['HEALTHY']:>4}  ({counts['HEALTHY']/max(total,1)*100:.1f}%)")
print(f"  Malades   : {counts['INFECTED']:>4}  ({counts['INFECTED']/max(total,1)*100:.1f}%)")
print(f"  Incertains: {counts['UNCERTAIN']:>3}  ({counts['UNCERTAIN']/max(total,1)*100:.1f}%)")

if results:
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResultats exportes vers: {args.output}")

print("Termine.\n")
