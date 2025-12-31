import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
model = models.efficientnet_b0(weights="DEFAULT").to(device) 
cross_entropy_loss = nn.CrossEntropyLoss() # creats the loss calculator
model.classifier[1] = torch.nn.Linear(1280,2).to(device)

if os.path.exists("iphone_classifier.pth"):
    model.load_state_dict(torch.load("iphone_classifier.pth"))


for param in model.parameters():
    param.requires_grad = False    # freeze all the weights dont need to retrain those

# unfreeze ONLY the classifier
for param in model.classifier.parameters():
    param.requires_grad = True


optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

dataset = datasets.ImageFolder('Apple Products',transform=data_transform)
data_loader = DataLoader(dataset,batch_size=32,shuffle=True)

num_epochs = 10  #In machine learning, an epoch is one complete pass through the entire training dataset
model.train()


for epoch in range(num_epochs):
    epoch_lost = 0
    for images, labels in data_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) #returns a tensor shaped [32, 2]
        loss = cross_entropy_loss(outputs,labels)
        epoch_lost += loss.item()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_lost/len(data_loader):.4f}")

model.eval()

test_image = Image.open('Apple Products/iPhone/iphonexr.jpg')


test_tensor = data_transform(test_image)
        

test_tensor = test_tensor.unsqueeze(0).to(device)


with torch.no_grad():
    output = model(test_tensor)
    predicted_class = output.argmax().item() #converts tensor to python value
    print(f"Predicted class: {dataset.classes[predicted_class]}")


torch.save(model.state_dict(), "iphone_classifier.pth")