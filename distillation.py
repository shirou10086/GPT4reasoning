import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# clip_model = load_clip_model()

class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 512)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

small_model = SmallModel()

# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(small_model.parameters(), lr=0.001)

for epoch in range(10):
    for images, texts in data_loader:

        with torch.no_grad():
            image_features_clip = clip_model.encode_image(images)
            text_features_clip = clip_model.encode_text(texts)


        image_features_small = small_model(images)

        loss = criterion(image_features_small, image_features_clip.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("finish")
