"""
Évaluation du modèle sur le dossier test/
Usage:
    python evaluate.py
    python evaluate.py --model 1 --folder test/
    python evaluate.py --both        # Compare les deux modèles
"""

import cv2
import numpy as np
import argparse
import os
import time
from tensorflow.keras.models import load_model

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL = True
except ImportError:
    MPL = False
    print("⚠️  matplotlib non disponible – pas de visualisation graphique.")

parser = argparse.ArgumentParser()
parser.add_argument("--model",  type=int, choices=[1, 2], default=2)
parser.add_argument("--folder", default="test")
parser.add_argument("--both",   action="store_true", help="Comparer les deux modèles")
parser.add_argument("--threshold-high", type=float, default=0.75)
parser.add_argument("--threshold-low",  type=float, default=0.25)
args = parser.parse_args()

MODEL_PATHS = {1: "peacock_model1.h5", 2: "peacock_model2.h5"}
MODEL_NAMES = {1: "CNN Basique (~85%)", 2: "MobileNetV2 (~95%)"}
IMG_SIZE    = 224
FORMATS     = (".jpg", ".jpeg", ".png", ".bmp")

def load_m(num):
    p = MODEL_PATHS[num]
    if not os.path.exists(p):
        print(f"❌ Modèle {p} introuvable."); return None
    print(f"  Chargement modèle {num}…")
    return load_model(p)

def predict_one(img_bgr, mdl, thr_high, thr_low):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)) / 255.0
    conf = float(mdl.predict(np.expand_dims(img, 0), verbose=0)[0][0])
    if conf > thr_high:   return "INFECTED", conf, "#e74c3c"
    elif conf < thr_low:  return "HEALTHY",  conf, "#27ae60"
    return "UNCERTAIN", conf, "#f39c12"

def run_eval(model_num, mdl, folder):
    images = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
              if f.lower().endswith(FORMATS)]
    if not images:
        print(f"  Aucune image dans '{folder}'."); return [], {}

    counts  = {"INFECTED": 0, "HEALTHY": 0, "UNCERTAIN": 0}
    records = []
    t0 = time.time()

    print(f"\n  Analyse de {len(images)} image(s)…\n")
    header = f"{'Fichier':<40} {'Statut':<12} {'Confiance':>9}"
    print("  " + header)
    print("  " + "-" * len(header))

    for path in images:
        img = cv2.imread(path)
        if img is None: continue
        status, conf, color = predict_one(img, mdl, args.threshold_high, args.threshold_low)
        counts[status] += 1
        icon = {"INFECTED": "MALADE  ", "HEALTHY": "SAIN    ", "UNCERTAIN": "INCERT. "}[status]
        fname = os.path.basename(path)
        print(f"  {fname:<40} {icon:<12} {conf:>8.2%}")
        records.append((path, status, conf, color))

    elapsed = time.time() - t0
    total = len(records)
    print(f"\n  Temps total : {elapsed:.2f}s  |  {elapsed/max(total,1)*1000:.0f}ms/image")
    return records, counts

def print_summary(model_num, counts):
    total = sum(counts.values())
    print(f"\n  ┌{'─'*36}┐")
    print(f"  │  RÉSUMÉ — Modèle {model_num}: {MODEL_NAMES[model_num]:<14}│")
    print(f"  ├{'─'*36}┤")
    print(f"  │  Total analysé : {total:<17}│")
    pct = lambda k: f"{counts[k]/max(total,1)*100:.1f}%"
    print(f"  │  🟢 Saines     : {counts['HEALTHY']:<6} ({pct('HEALTHY'):<6})        │")
    print(f"  │  🔴 Malades    : {counts['INFECTED']:<6} ({pct('INFECTED'):<6})        │")
    print(f"  │  🟡 Incertains : {counts['UNCERTAIN']:<6} ({pct('UNCERTAIN'):<6})        │")
    print(f"  └{'─'*36}┘")

def visualize(records, model_num):
    if not MPL or not records: return
    n = min(len(records), 9)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.patch.set_facecolor('#0d0f1a')
    axes = np.array(axes).flatten() if rows > 1 or cols > 1 else [axes]

    for i, (path, status, conf, color) in enumerate(records[:n]):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax  = axes[i]
        ax.imshow(img)
        ax.axis('off')
        label = {"INFECTED": "Malade", "HEALTHY": "Saine", "UNCERTAIN": "Incertain"}[status]
        ax.set_title(f"{os.path.basename(path)}\n{label} — {conf:.1%}",
                     color=color, fontsize=8, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

    for ax in axes[n:]: ax.axis('off')

    fig.suptitle(f"Résultats — Modèle {model_num}: {MODEL_NAMES[model_num]}",
                 color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = f"evaluation_model{model_num}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d0f1a')
    print(f"\n  📊 Planche visuelle sauvegardée : {out_path}")
    plt.show()

# ── Vérification dossier ───────────────────────────────────
if not os.path.isdir(args.folder):
    print(f"❌ Dossier '{args.folder}' introuvable."); exit(1)

models_to_run = [1, 2] if args.both else [args.model]
print("=" * 60)
print(f"  🌿 ÉVALUATION — Dossier : {args.folder}/")
print("=" * 60)

for mnum in models_to_run:
    print(f"\n▶  Modèle {mnum} : {MODEL_NAMES[mnum]}")
    mdl = load_m(mnum)
    if mdl is None: continue
    records, counts = run_eval(mnum, mdl, args.folder)
    print_summary(mnum, counts)
    visualize(records, mnum)

print("\n✅ Évaluation terminée.\n")
