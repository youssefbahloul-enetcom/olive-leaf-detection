# 🌿 Olive Leaf Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Detect "Peacock Eye" disease (œil de paon) on olive leaves using Deep Learning

## 📌 Description

This project uses a Convolutional Neural Network (CNN) to detect the "Peacock Eye" disease (*Spilocaea oleagina*) on olive leaves. The model classifies leaves as either **Healthy** or **Infected** (Peacock Eye).

The project includes two models :
- **Model 1**: Basic CNN architecture (simple Conv2D layers)
- **Model 2**: Transfer Learning with MobileNetV2 (recommended - better accuracy)

## 🧠 Technologies

- **Python** 3.8+
- **TensorFlow / Keras** - Deep Learning framework
- **OpenCV** - Image processing
- **MobileNetV2** - Pre-trained model for transfer learning

## 📊 Results

| Model | Validation Accuracy |
|-------|---------------------|
| Model 1 (Basic CNN) | ~85% |
| Model 2 (MobileNetV2) | ~95% |

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/olive-disease-detection.git
cd olive-disease-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the real-time detection:
```bash
python run_projet.py
```

Press `q` to quit the application.

### Using the model in your own code:
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("peacock_model2.h5")

# Prepare image
img = cv2.resize(image, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img, verbose=0)
confidence = prediction[0][0]

if confidence > 0.75:
    result = "Infected: Peacock Eye"
elif confidence < 0.25:
    result = "Healthy"
```

## 📁 Project Structure

```
olive-disease-detection/
├── README.md                 # This file
├── requirements.txt         # Python dependencies
├── LICENSE                 # MIT License
├── run_projet.py            # Main script for real-time detection
├── peacock_model2.h5       # Trained model (MobileNetV2)
├── peacock_model1.h5       # Trained model (Basic CNN)
├── oeil_de_paon(model1).ipynb    # Model 1 training notebook
├── oirl_de_paon(model2).ipynb   # Model 2 training notebook
└── TODO.md                 # Development tasks
```

## 🔬 Model Details

### Model 2 Architecture (Recommended)
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Feature Extraction**: Global Average Pooling
- **Dense Layers**: 128 units with ReLU activation
- **Dropout**: 0.5 for regularization
- **Output**: Sigmoid activation for binary classification

### Training Parameters
- **Image Size**: 224x224
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔧 Project Structure

```
olive-disease-detection/
├── .gitignore              # Git ignore rules
├── LICENSE                # MIT License
├── README.md               # This file
├── requirements.txt       # Python dependencies
├── run_projet.py           # Main script for real-time detection
├── peacock_model1.h5       # Trained model (Basic CNN)
├── peacock_model2.h5       # Trained model (MobileNetV2)
├── oeil_de_paon(model1).ipynb    # Model 1 training notebook
├── oirl_de_paon(model2).ipynb   # Model 2 training notebook
├── data_olive_peacock_spot/     # Dataset folder
├── test/                   # Test images
└── TODO.md                 # Development tasks
```

## 🙏 Acknowledgments

- Dataset source: [Describe your dataset source]
- Inspired by: [Any inspiring projects or papers]

---

⭐ If you find this project useful, please give it a star!
