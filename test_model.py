import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network model
class SimpleFashionNet(torch.nn.Module):
    def __init__(self):
        super(SimpleFashionNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def process_image(image_path):
    image = Image.open(image_path).convert('L')
    resize_transform = Resize((28, 28))
    image = resize_transform(image)
    transform = ToTensor()
    tensor = transform(image)
    return tensor

def predict_image(image_path, model, class_labels):
    image_tensor = process_image(image_path)
    with torch.no_grad():
        model.eval()
        outputs = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, 1).item()
    return predicted_class, probabilities.squeeze().tolist()

if __name__ == "__main__":
    model = SimpleFashionNet()
    model.load_state_dict(torch.load('fashion_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # Load class labels for reference
    class_labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # Provide the path to the image of your choice
    image_path = 'photos/pants.jpg'  # Replace with the actual path to your test image

    # Make a prediction
    predicted_class, probabilities = predict_image(image_path, model, class_labels)

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
