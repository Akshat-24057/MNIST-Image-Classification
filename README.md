# MNIST Digit Classification with CNN

A comprehensive deep learning project for handwritten digit classification using Convolutional Neural Networks (CNN) implemented in PyTorch on the MNIST dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Output Files](#output-files)
- [Results](#results)
- [Customization](#customization)

## 🔍 Overview

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits (0-9) from the MNIST dataset. The implementation includes data exploration, model design, training, and comprehensive evaluation with visualizations.

## ✨ Features

- **Automatic GPU/CPU Detection**: Utilizes CUDA if available for faster training
- **Comprehensive Data Exploration**:
  - Sample image visualization
  - Class distribution analysis
  - Dataset statistics
- **Custom CNN Architecture**:
  - Two convolutional layers with ReLU activation
  - Max pooling for dimensionality reduction
  - Fully connected layers with dropout regularization
- **Advanced Training**:
  - Training/validation accuracy and loss tracking
  - Progress monitoring per epoch
  - Early insight into model convergence
- **Detailed Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix visualization
  - Misclassification analysis
- **Visualization Suite**:
  - Sample images from dataset
  - Class distribution charts
  - Training curves (loss and accuracy)
  - Confusion matrix heatmap
- **Model Persistence**: Save trained model for future use

## 📦 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.19.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

## 🚀 Installation

1. Clone or download this repository

2. Install PyTorch (visit [PyTorch.org](https://pytorch.org/) for installation instructions specific to your system)

3. Install other required packages:
```bash
pip install torchvision matplotlib numpy seaborn scikit-learn
```

4. For GPU support, ensure CUDA is properly installed and PyTorch is built with CUDA support

## 📊 Dataset

### MNIST Dataset
- **Training Set**: 60,000 grayscale images (28×28 pixels)
- **Test Set**: 10,000 grayscale images (28×28 pixels)
- **Classes**: 10 (digits 0-9)
- **Auto-Download**: The script automatically downloads MNIST data on first run

The dataset will be downloaded to a `./data` directory in your project folder.

## 🏗️ Model Architecture

### CNNModel

```
Input: [batch_size, 1, 28, 28]
    ↓
Conv2D(1→32, kernel=3×3, padding=1) + ReLU + MaxPool(2×2)
    ↓ [batch_size, 32, 14, 14]
Conv2D(32→64, kernel=3×3, padding=1) + ReLU + MaxPool(2×2)
    ↓ [batch_size, 64, 7, 7]
Flatten
    ↓ [batch_size, 3136]
Fully Connected(3136→128) + ReLU + Dropout(0.5)
    ↓ [batch_size, 128]
Fully Connected(128→10)
    ↓ [batch_size, 10]
Output: Logits for 10 classes
```

### Architecture Details:
- **Convolutional Layers**: Extract spatial features
  - Layer 1: 32 filters, 3×3 kernel
  - Layer 2: 64 filters, 3×3 kernel
- **Pooling Layers**: Reduce spatial dimensions by 2×
- **Activation**: ReLU for non-linearity
- **Regularization**: 50% dropout to prevent overfitting
- **Output**: 10 neurons (one per digit class)

### Parameters:
- **Total Parameters**: ~122,000
- **Batch Size**: 100
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy

## 💻 Usage

### Basic Usage

1. Run the script:
```bash
python image_classification.py
```

2. The script will automatically:
   - Download MNIST dataset (if not already present)
   - Perform exploratory data analysis
   - Build and display the CNN model
   - Train the model for 15 epochs
   - Evaluate performance on test set
   - Generate visualizations
   - Save the trained model

### Expected Output
```
Using device: cuda  # or cpu
Performing Exploratory Data Analysis...
Class Distribution:
Class 0: 5923 samples (9.87%)
...
Starting model training...
Epoch [1/15], Train Loss: 0.2145, Train Acc: 93.45%, Val Loss: 0.0523, Val Acc: 98.21%
...
Model Evaluation Metrics:
Accuracy: 0.9912
Precision: 0.9911
Recall: 0.9912
F1-Score: 0.9911
```

## 🎓 Training

### Hyperparameters
```python
batch_size = 100        # Images per batch
learning_rate = 0.001   # Adam optimizer learning rate
num_epochs = 15         # Training iterations
```

### Training Process
1. **Forward Pass**: Images through CNN
2. **Loss Calculation**: Cross-entropy between predictions and labels
3. **Backward Pass**: Compute gradients
4. **Optimization**: Update weights using Adam
5. **Validation**: Evaluate on test set after each epoch

### Training Monitoring
- Real-time epoch-by-epoch metrics
- Training and validation loss tracking
- Training and validation accuracy tracking

## 📈 Evaluation

### Metrics Computed
- **Accuracy**: Overall classification correctness
- **Precision**: Correctness of positive predictions (weighted average)
- **Recall**: Coverage of actual positives (weighted average)
- **F1-Score**: Harmonic mean of precision and recall

### Confusion Matrix Analysis
- 10×10 matrix showing true vs predicted labels
- Identifies which digits are commonly confused
- Highlights misclassification patterns

### Misclassification Analysis
The script identifies:
- Top 3 most commonly misclassified digits
- Most frequent confusion pairs (e.g., 4 predicted as 9)
- Misclassification rates per digit

## 📤 Output Files

### 1. `sample_images.png`
Random sample of 5 images from the training dataset with their labels

### 2. `class_distribution.png`
Bar chart showing the frequency of each digit (0-9) in the training set

### 3. `training_curves.png`
Two subplots:
- Training and validation loss over epochs
- Training and validation accuracy over epochs

### 4. `confusion_matrix.png`
Heatmap showing the 10×10 confusion matrix for test set predictions

### 5. `mnist_cnn_model.pth`
Saved model weights for future use (inference or continued training)

## 📊 Results

### Expected Performance
With the default configuration, the model typically achieves:
- **Test Accuracy**: ~99%
- **Training Time**: ~5-10 minutes (GPU) or ~30-45 minutes (CPU)

### Common Confusions
The most commonly confused digit pairs are usually:
- 4 ↔ 9
- 3 ↔ 5
- 7 ↔ 1
- 2 ↔ 7

## 🔧 Customization

### Adjust Hyperparameters
```python
# Change batch size
batch_size = 64  # Smaller batch for more frequent updates

# Adjust learning rate
learning_rate = 0.0001  # Lower learning rate for finer tuning

# Train for more epochs
num_epochs = 25  # More epochs for better convergence
```

### Modify Model Architecture
```python
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Add a third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Adjust dropout rate
        self.dropout = nn.Dropout(0.3)  # Less aggressive dropout
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
```

### Add Data Augmentation
```python
transform = transforms.Compose([
    transforms.RandomRotation(10),      # Rotate images
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Slight translation
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Change Optimizer
```python
# Use SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Use RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

### Add Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# In training loop, after each epoch:
scheduler.step()
```

## 💾 Loading Saved Model

To load and use the trained model:

```python
# Load the model
model = CNNModel().to(device)
model.load_state_dict(torch.load('mnist_cnn_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
```

## 🧪 Testing with Custom Images

```python
from PIL import Image

def predict_custom_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()
```

## 🎯 Performance Tips

### Improve Accuracy
1. **Increase Model Capacity**: Add more convolutional layers or filters
2. **Data Augmentation**: Apply random transformations to training data
3. **Batch Normalization**: Add BatchNorm layers for stable training
4. **Learning Rate Tuning**: Experiment with different learning rates
5. **Ensemble Methods**: Combine multiple models

### Speed Up Training
1. **Use GPU**: Ensure CUDA is available and utilized
2. **Increase Batch Size**: If memory allows, larger batches are faster
3. **Mixed Precision Training**: Use `torch.cuda.amp` for faster training
4. **Efficient Data Loading**: Use `num_workers` in DataLoader

```python
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster data transfer to GPU
)
```

## 📝 Notes

- The model uses normalized inputs (mean=0.1307, std=0.3081) based on MNIST statistics
- Dropout is only active during training, automatically disabled during evaluation
- The confusion matrix helps identify specific digits that need more training attention
- Class distribution is balanced in MNIST, so accuracy is a reliable metric

## 🤝 Contributing

Areas for improvement:
- Implement advanced architectures (ResNet, DenseNet)
- Add cross-validation
- Implement ensemble methods
- Add real-time inference demo with webcam
- Create web interface for digit recognition

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 32  # or even smaller
```

### Slow Training on CPU
```python
# Reduce model complexity or use GPU
# Alternatively, reduce number of epochs for testing
num_epochs = 5
```

### Poor Convergence
```python
# Try different learning rate
learning_rate = 0.01  # or 0.0001

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
```

### Model Overfitting
```python
# Increase dropout
self.dropout = nn.Dropout(0.7)

# Add data augmentation
# Reduce model capacity
```

## 📚 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)

## 📄 License

This project is available for educational and research purposes.

---

**Author**: [Your Name]  
**Last Updated**: 2026  
**Contact**: [Your Email]

**Hardware Recommendations**:
- **CPU**: Training takes ~30-45 minutes
- **GPU** (NVIDIA with CUDA): Training takes ~5-10 minutes
- **RAM**: Minimum 4GB, Recommended 8GB
