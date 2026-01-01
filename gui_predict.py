import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os

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

class CarBikeClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗🚲 Car/Bike Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Charger le modèle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN().to(self.device)
        
        try:
            checkpoint = torch.load('car_bike_model_full.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classes = checkpoint['classes']
            self.img_size = checkpoint['img_size']
        except:
            # Fallback si le fichier full n'existe pas
            self.model.load_state_dict(torch.load('car_bike_model.pth', map_location=self.device))
            self.classes = ['bike', 'car']
            self.img_size = 128
        
        self.model.eval()
        
        # Transformation pour les images
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.current_image_path = None
        self.setup_ui()
    
    def setup_ui(self):
        # Titre
        title_label = tk.Label(
            self.root,
            text="🚗 Car vs Bike Classifier 🚲",
            font=("Arial", 24, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=20)
        
        # Bouton pour charger une image
        self.upload_btn = tk.Button(
            self.root,
            text="📁 Upload Image",
            font=("Arial", 14, "bold"),
            bg='#3498db',
            fg='white',
            padx=30,
            pady=10,
            command=self.upload_image,
            cursor='hand2'
        )
        self.upload_btn.pack(pady=10)
        
        # Frame pour l'image
        self.image_frame = tk.Frame(self.root, bg='#34495e', width=400, height=300)
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="No image loaded",
            font=("Arial", 12),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.image_label.pack(expand=True)
        
        # Label pour la prédiction
        self.prediction_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 20, "bold"),
            bg='#2c3e50',
            fg='#2ecc71'
        )
        self.prediction_label.pack(pady=20)
        
        # Label pour la confiance
        self.confidence_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 14),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.confidence_label.pack()
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_image(file_path)
    
    def display_image(self, image_path):
        try:
            # Charger et redimensionner l'image pour l'affichage
            img = Image.open(image_path)
            img.thumbnail((380, 280), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Garder une référence
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def predict_image(self, image_path):
        try:
            # Charger et transformer l'image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.classes[predicted.item()]
                confidence_percent = confidence.item() * 100
            
            # Afficher le résultat
            emoji = "🚲" if predicted_class == "bike" else "🚗"
            self.prediction_label.configure(
                text=f"{emoji} Prediction: {predicted_class.upper()} {emoji}",
                fg='#2ecc71' if confidence_percent > 80 else '#f39c12'
            )
            self.confidence_label.configure(
                text=f"Confidence: {confidence_percent:.2f}%"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.prediction_label.configure(text="❌ Prediction failed")
            self.confidence_label.configure(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = CarBikeClassifierGUI(root)
    root.mainloop()
