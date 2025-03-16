from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import gdown
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# 🔹 Google Drive model download
MODEL_URL = "https://drive.google.com/file/d/1a4CRfgWJ1Rw5zogLKEVlXaZoI-SFdVwI/view?usp=drive_link"
MODEL_PATH = "best_model.pth"

# Check if model exists, otherwise download
try:
    print("📥 Checking for model file...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading model: {e}")

# 🔹 Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 34)  # Adjust for number of classes

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 🔹 Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/upload', methods=['POST'])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        # Process Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Model Prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)
            predicted_label = f"Component {predicted_class.item()}"

        return jsonify({"component": predicted_label, "description": "Detailed description of component."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)