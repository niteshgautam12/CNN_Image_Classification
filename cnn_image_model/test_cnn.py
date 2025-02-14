import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from cnn_classifier import CNNModel  # Import your trained model class

#  Load the trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.eval()

# Define image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to CIFAR-10 dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load and preprocess your custom image
# Load and preprocess the image
image_path = "my_image.jpg"  # Keep PNG format
image = Image.open(image_path).convert("RGB")  # Convert to RGB (removes Alpha channel)

# Apply transformations
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension


#  Perform inference
with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)  # Get class with highest probability

# Define CIFAR-10 class labels
class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#  Show the result
predicted_label = class_labels[predicted_class.item()]
print(f"Predicted Class: {predicted_label}")

#  Display the image
plt.imshow(Image.open('my_image.jpg'))
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()

