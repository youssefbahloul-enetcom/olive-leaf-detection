# Olive Leaf Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Detect "Peacock Eye" disease (*Spilocaea oleagina*) on olive leaves using Deep Learning — with real-time webcam detection and a full web interface.

## Description

This project uses Convolutional Neural Networks to classify olive leaves as **Healthy** or **Infected** (Peacock Eye / Oeil de Paon). It includes two trained models, a real-time webcam detector, a web interface for image upload, a batch prediction tool, and an evaluation script.

## Models

| Model | Architecture | Validation Accuracy |
|-------|-------------|---------------------|
| Model 1 | Basic CNN (Conv2D from scratch) | ~85% |
| Model 2 | Transfer Learning — MobileNetV2 | ~95% ✅ Recommended |

## Technologies

- **Python** 3.8+
- **TensorFlow / Keras** — Deep Learning
- **MobileNetV2** — Transfer Learning
- **OpenCV** — Image processing & webcam
- **Flask** — Web interface
- **NumPy / Matplotlib** — Data processing & visualization
- **scikit-learn** — Evaluation metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/olive-disease-detection.git
cd olive-disease-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface (recommended)

Launch the Flask web application and open it in your browser:

```bash
python app.py
```

Then navigate to `http://localhost:5000`.

Features:
- Drag and drop image upload
- Model selection (Model 1 or Model 2)
- Adjustable confidence thresholds via sliders
- Animated confidence gauge
- Session history of the last 10 predictions

---

### Real-Time Webcam Detection

```bash
python run_projet.py                    # Model 2 by default
python run_projet.py --model 1          # Use basic CNN
python run_projet.py --camera 1         # Use external camera
python run_projet.py --threshold-high 0.8 --threshold-low 0.2
```

Keyboard shortcuts during detection:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot to `screenshots/` folder |
| `m` | Switch between models |
| `p` | Pause / Resume detection |

---

### Static Image Prediction

Run predictions on a single image or an entire folder:

```bash
python predict_image.py --input test/
python predict_image.py --input my_leaf.jpg --show
python predict_image.py --input test/ --model 1 --output results.csv
```

Results are exported to a CSV file automatically.

---

### Evaluation Script

Evaluate model performance on a folder of images:

```bash
python evaluate.py                      # Model 2 on test/ folder
python evaluate.py --model 1
python evaluate.py --both               # Compare both models
python evaluate.py --folder data_olive_peacock_spot
```

Outputs a terminal report and saves a visual prediction montage as a PNG file.

## Project Structure

```
olive-disease-detection/
├── app.py                        # Flask web application
├── run_projet.py                 # Real-time webcam detection
├── predict_image.py              # Static image / batch prediction
├── evaluate.py                   # Model evaluation and reporting
├── requirements.txt              # Python dependencies
├── LICENSE
├── README.md
├── templates/
│   └── index.html                # Web interface frontend
├── peacock_model1.h5             # Trained Model 1 (Basic CNN)
├── peacock_model2.h5             # Trained Model 2 (MobileNetV2)
├── oeil_de_paon(model1).ipynb    # Training notebook — Model 1
├── oirl_de_paon(model2).ipynb    # Training notebook — Model 2
├── data_olive_peacock_spot/      # Dataset
├── test/                         # Test images
└── screenshots/                  # Auto-created by webcam script
```

## Model Details

### Model 2 — MobileNetV2 (Recommended)

- **Base**: MobileNetV2 pre-trained on ImageNet
- **Head**: Global Average Pooling → Dense (128, ReLU) → Dropout (0.5) → Sigmoid
- **Input size**: 224 × 224
- **Optimizer**: Adam — Learning rate: 0.0001
- **Loss**: Binary Crossentropy

## Future Improvements

- Add a "no leaf detected" class for improved robustness
- Integrate object detection before classification
- Deploy as a cloud API (Docker + REST)
- Mobile application

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

If you find this project useful, please consider giving it a star on GitHub.
