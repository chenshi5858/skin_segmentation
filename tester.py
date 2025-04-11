import torch
import numpy as np
from PIL import Image
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)  
        self.fc2 = nn.Linear(4, 8)  
        self.fc3 = nn.Linear(8, n_classes)  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

model = SimpleModel(3, 1)
model.load_state_dict(torch.load("skin_segmentation.pth"))
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  
    image = np.asarray(image, dtype=np.float32) / 255.0  
    return image

def classify_image(image_array):
     height, width, _ = image_array.shape
     pixels = image_array.reshape(-1, 3)  
     pixel_tensors = torch.tensor(pixels, dtype=torch.float32)  
     output = model(pixel_tensors).detach().numpy() 
     output = output.reshape(height, width) 
     output = (output > 0.5) * 255  
     return output.astype(np.uint8)

image_path = "persona.jpeg"  
image_array = preprocess_image(image_path)
classified_image = classify_image(image_array)

Image.fromarray(classified_image.astype(np.uint8)).save("output.png")

print("Clasificación completada. Imagen guardada como output.png")
