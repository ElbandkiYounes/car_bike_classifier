# 🚗🚲 Car vs Bike Classifier

A deep learning image classifier built with PyTorch that distinguishes between cars and bikes. Features both a web interface (Gradio) and desktop GUI (Tkinter) for easy image classification.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Features

- **Deep Learning CNN Model** - Custom 3-layer convolutional neural network
- **High Accuracy** - 99.01% training accuracy, 97.83% validation accuracy
- **Dual Interface** - Both web-based and desktop applications
- **Real-time Prediction** - Instant classification with confidence scores
- **Easy to Use** - Simple drag-and-drop or upload functionality

## 🏗️ Project Structure

```
car_bike_classifier/
│
├── dataset/
│   ├── train/
│   │   ├── bike/       # Training bike images
│   │   └── car/        # Training car images
│   └── test/
│       ├── bike/       # Test bike images
│       └── car/        # Test car images
│
├── train_torch.py      # Training script (PyTorch)
├── web_predict.py      # Web interface (Gradio)
├── gui_predict.py      # Desktop GUI (Tkinter)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/car_bike_classifier.git
cd car_bike_classifier
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

**Activate the virtual environment:**

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

- **Linux/Mac:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset Setup

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── bike/    # Add bike training images here
│   └── car/     # Add car training images here
└── test/
    ├── bike/    # Add bike test images here
    └── car/     # Add car test images here
```

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`

## 🚀 Training the Model

Run the training script to create your model:

```bash
python train_torch.py
```

**Training output:**
- Model will train for 10 epochs (configurable in script)
- Progress displayed with loss and accuracy metrics
- Trained model saved as `car_bike_model.pth` and `car_bike_model_full.pth`

**Training parameters (customizable in `train_torch.py`):**
- Image size: 128x128
- Batch size: 16
- Epochs: 10
- Learning rate: 0.001

## 🖥️ Running the Application

### Option 1: Web Interface (Recommended)

Launch the Gradio web interface:

```bash
python web_predict.py
```

Then open your browser to: **http://127.0.0.1:7860**

**Features:**
- ✅ Works in any browser
- ✅ Clean, modern UI
- ✅ Real-time predictions
- ✅ Confidence scores
- ✅ Can be shared online with `share=True`

### Option 2: Desktop GUI

**Note:** Requires Tkinter (usually pre-installed with Python)

```bash
python gui_predict.py
```

**Features:**
- ✅ Native desktop application
- ✅ Drag-and-drop interface
- ✅ Fast predictions
- ✅ Works offline

**Troubleshooting Tkinter:**
If you encounter Tkinter errors, use your system Python instead:
```bash
deactivate  # Exit virtual environment
python gui_predict.py
```

## 📈 Model Architecture

```
SimpleCNN(
  ├── Conv2D(3 → 32) + ReLU + MaxPool
  ├── Conv2D(32 → 64) + ReLU + MaxPool
  ├── Conv2D(64 → 128) + ReLU + MaxPool
  ├── Flatten
  ├── Dense(128) + ReLU + Dropout(0.5)
  └── Dense(2) [Output: bike, car]
)
```

**Performance:**
- Training Accuracy: **99.01%**
- Validation Accuracy: **97.83%**
- Best Epoch: **100%** accuracy on test set

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Pillow
- Gradio (for web interface)
- Tkinter (for desktop GUI, usually pre-installed)

See `requirements.txt` for complete list.

## 📝 Usage Example

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load('car_bike_model.pth'))
model.eval()

# Transform image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
img = Image.open('test_image.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)
output = model(img_tensor)
prediction = torch.argmax(output, dim=1)
```

