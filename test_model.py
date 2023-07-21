import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
class FashionNet(torch.nn.Module):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FashionNet()
model.load_state_dict(torch.load('fashion_model.pth'))
model.eval()

# Define a function to process the input image
def process_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    resize_transform = Resize((28, 28))  # Resize the image to 28x28 pixels
    image = resize_transform(image)
    transform = ToTensor()
    tensor = transform(image)
    return tensor

# Define a function to make predictions
def predict_image(image_path, model):
    image_tensor = process_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), probabilities.squeeze().tolist()

# Provide the path to the image of your choice
image_path = 'shirt.jpg'  # Replace with the actual path to your test image

# Make a prediction
predicted_class, probabilities = predict_image(image_path, model)

# Load class labels for reference
class_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Display the results
print(f'Predicted class: {class_labels[predicted_class]}')
print('Probabilities:')
for i, prob in enumerate(probabilities):
    print(f'{class_labels[i]}: {prob:.5f}')

# Display the image
img = Image.open(image_path)
plt.imshow(img)
plt.title(f'Predicted class: {class_labels[predicted_class]}')
plt.axis('off')
plt.show()
