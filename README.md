# 🛰️ EuroSAT Land Use Classification with CNN

A deep learning project that classifies satellite images into 10 different land use categories using Convolutional Neural Networks (CNN). Achieved **97.90% test accuracy** on the EuroSAT dataset.

![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.90%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a custom Convolutional Neural Network to automatically classify satellite imagery from the EuroSAT dataset. The model distinguishes between 10 different land use and land cover classes, demonstrating the power of deep learning in remote sensing applications.

**Key Achievements:**
- 97.90% accuracy on test set
- Perfect classification (100%) on Residential and SeaLake categories
- Robust performance across all 10 land use classes
- No significant overfitting observed

## 📊 Dataset

**EuroSAT RGB Dataset**
- **Total Images:** 27,000 labeled satellite images
- **Image Size:** 64×64 pixels, RGB (3 channels)
- **Classes:** 10 land use categories
- **Source:** Sentinel-2 satellite imagery

### Land Use Categories:
1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

### Data Split:
- **Training Set:** 18,900 images (70%)
- **Validation Set:** 4,050 images (15%)
- **Test Set:** 4,050 images (15%)

## 🏗️ Model Architecture

```
Input Layer: 64×64×3 RGB images
    ↓
Conv2D (32 filters, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    ↓
Conv2D (64 filters, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    ↓
Conv2D (128 filters, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    ↓
Conv2D (256 filters, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    ↓
Flatten (1,024 neurons)
    ↓
Dense (256) + ReLU + Dropout(0.5)
    ↓
Dense (128) + ReLU + Dropout(0.3)
    ↓
Output Layer: Dense (10) + Softmax
```

**Total Parameters:** 686,922 (2.62 MB)

### Key Features:
- **Batch Normalization:** Stabilizes training and accelerates convergence
- **Dropout Layers:** Prevents overfitting (0.5 and 0.3 dropout rates)
- **Progressive Filters:** Increases from 32 to 256 filters to capture hierarchical features
- **Data Augmentation:** Random flips and rotations for improved generalization

## 📈 Results

### Overall Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | **97.90%** |
| Test Loss | 0.0693 |
| Correct Predictions | 3,965 / 4,050 |
| Misclassifications | 85 |

### Per-Class Accuracy
| Class | Accuracy |
|-------|----------|
| Residential | 100.00% |
| SeaLake | 100.00% |
| Forest | 98.76% |
| Industrial | 98.72% |
| AnnualCrop | 96.38% |
| HerbaceousVegetation | 94.77% |
| PermanentCrop | 96.76% |
| Highway | 98.35% |
| River | 97.37% |
| Pasture | 97.71% |

### Training Insights
- **Training Time:** ~3.5 hours (50 epochs)
- **Best Epoch:** 47
- **Learning Rate Schedule:** Adaptive reduction at epochs 20 and 43
- **Convergence:** Achieved stable performance with minimal validation fluctuation

### Visualizations

**Training History:**
![Training History](images/SatelliteProjectAccuracyLoss.png)

**Confusion Matrix:**
![Confusion Matrix](images/SatelliteProjectConfusionMatrix.png)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BradenWPeterson/EuroSATCNN.git
cd EuroSATCNN
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install tensorflow tensorflow-datasets numpy scikit-learn matplotlib seaborn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

```python
import tensorflow_datasets as tfds
from model import create_cnn_model, train_model

# Load dataset
ds, info = tfds.load('eurosat/rgb', with_info=True, as_supervised=True)

# Create and train model
model = create_cnn_model()
history = train_model(model, ds)
```

### Making Predictions

```python
from tensorflow import keras
import numpy as np

# Load the trained model
model = keras.models.load_model('best_model.keras')

# Load and preprocess your image
image = load_and_preprocess_image('path/to/image.jpg')

# Make prediction
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_class = np.argmax(prediction)

class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']

print(f"Predicted class: {class_names[predicted_class]}")
print(f"Confidence: {np.max(prediction)*100:.2f}%")
```

### Evaluating the Model

```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
```

## 📁 Project Structure

```
eurosat-classifier/
├── index.html                  # Portfolio website
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── notebooks/
│   └── eurosat_cnn.ipynb      # Jupyter notebook with full workflow
└── images/
    ├── training_history.png   # Training curves
    └── confusion_matrix.png   # Confusion matrix visualization
```

## 🛠️ Technologies

- **TensorFlow/Keras** - Deep learning framework
- **TensorFlow Datasets** - Dataset loading and management
- **NumPy** - Numerical computations
- **Scikit-learn** - Evaluation metrics
- **Matplotlib/Seaborn** - Visualization
- **Python 3.8+** - Programming language

## 🔮 Future Improvements

- [ ] Implement transfer learning with pre-trained models (ResNet, EfficientNet)
- [ ] Add model interpretability with Grad-CAM visualizations
- [ ] Deploy as a web application with real-time predictions
- [ ] Experiment with different augmentation strategies
- [ ] Test on other satellite image datasets
- [ ] Create an ensemble model for improved accuracy
- [ ] Optimize model for mobile deployment with TensorFlow Lite

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EuroSAT Dataset:** Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
- **TensorFlow Team** for the excellent deep learning framework
- **Sentinel-2 Mission** for providing the satellite imagery

## 📞 Contact

Braden Peterson - [bradenwpeterson@gmail.com](mailto:bradenwpeterson@gmail.com)

Project Link: [https://github.com/BradenWPeterson/EuroSATCNN](https://github.com/BradenWPeterson/EuroSATCNN)

---

⭐ If you found this project helpful, please consider giving it a star!
