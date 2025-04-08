# app.py
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

app = Flask(__name__)

model = DigitCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['image'].split(",")[1]
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    img = Image.open(image_bytes).convert("L")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, dim=1).item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
