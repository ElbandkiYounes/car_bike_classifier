import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np

# Définition du modèle (même architecture que l'entraînement)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

try:
    checkpoint = torch.load('car_bike_model_full.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    classes = checkpoint['classes']
    img_size = checkpoint['img_size']
except:
    model.load_state_dict(torch.load('car_bike_model.pth', map_location=device))
    classes = ['bike', 'car']
    img_size = 128

model.eval()

# Transformation pour les images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    """
    Fait une prédiction sur l'image uploadée
    """
    if image is None:
        return None
    
    try:
        # Convertir l'image numpy en PIL Image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        # Transformer l'image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Créer un dictionnaire des probabilités pour chaque classe
            probs_dict = {
                classes[i]: float(probabilities[0][i]) 
                for i in range(len(classes))
            }
        
        return probs_dict
    
    except Exception as e:
        return None

# Créer l'interface Gradio
with gr.Blocks(title="🚗🚲 Car/Bike Classifier") as demo:
    gr.Markdown(
        """
        # 🚗 Car vs Bike Classifier 🚲
        
        Upload an image to classify whether it's a **car** or a **bike**!
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="numpy",
                height=400
            )
            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Label(
                label="Prediction Results",
                num_top_classes=2
            )
    
    # Exemples d'images (si disponibles)
    gr.Markdown("### 💡 Try with test images:")
    gr.Examples(
        examples=[
            ["dataset/test/bike/" + img for img in ["bike_1.jpg", "bike_2.jpg"]][:1] if True else [],
            ["dataset/test/car/" + img for img in ["car_1.jpg", "car_2.jpg"]][:1] if True else []
        ][:0],  # Désactivé par défaut
        inputs=image_input,
    )
    
    # Connecter le bouton à la fonction de prédiction
    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=output
    )
    
    # Permettre aussi la prédiction en changeant l'image
    image_input.change(
        fn=predict_image,
        inputs=image_input,
        outputs=output
    )
    
    gr.Markdown(
        """
        ---
        ### 📊 Model Information
        - **Architecture**: Simple CNN (3 convolutional layers)
        - **Training Accuracy**: 99.01%
        - **Validation Accuracy**: 97.83%
        - **Classes**: Bike, Car
        """
    )

# Lancer l'application
if __name__ == "__main__":
    print("🚀 Starting Car/Bike Classifier Interface...")
    print("📱 Open your browser and go to the URL shown below")
    demo.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=7860,
        css="""
        footer {display: none !important;}
        .gradio-container {min-height: 0px !important;}
        """
    )
