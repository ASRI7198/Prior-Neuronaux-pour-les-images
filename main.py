import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MyUNet(nn.Module):
    def __init__(self):
        super(MyUNet, self).__init__()
        self.down1 = nn.Conv2d(16, 32, 5, 2, 2)
        self.down2 = nn.Conv2d(32, 64, 5, 2, 2)
        self.down3 = nn.Conv2d(64, 128, 5, 2, 2)
        self.down4 = nn.Conv2d(128, 256, 5, 2, 2)

        self.up4 = nn.ConvTranspose2d(256, 124, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(128, 60, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

        self.skip1 = nn.Conv2d(128, 4, 5, 1, 2)
        self.skip2 = nn.Conv2d(64, 4, 5, 1, 2)

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        down1 = self.down1(x)
        down1 = self.leaky_relu(down1)

        down2 = self.down2(down1)
        down2 = self.leaky_relu(down2)

        down3 = self.down3(down2)
        down3 = self.leaky_relu(down3)

        down4 = self.down4(down3)
        down4 = self.leaky_relu(down4)

        up4 = self.up4(down4)
        up4 = self.leaky_relu(up4)

        skip1 = self.skip1(down3)
        skip1 = self.leaky_relu(skip1)
        up3 = self.up3(torch.cat([skip1, up4], dim=1))
        up3 = self.leaky_relu(up3)

        skip2 = self.skip2(down2)
        skip2 = self.leaky_relu(skip2)
        up2 = self.up2(torch.cat([skip2, up3], dim=1))
        up2 = self.leaky_relu(up2)

        up1 = self.up1(up2)
        up1 = self.sigmoid(up1)

        return up1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

myunet = MyUNet()
myunet.to(device)

input_img = Image.open('testcrop.jpg')
transform = transforms.Compose([transforms.PILToTensor()])
target = transform(input_img) / 255
target = target.unsqueeze(0).float().to(device)

h = target.size()[2]
w = target.size()[3]
z = torch.rand(1, 16, h, w).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(myunet.parameters(), lr=0.0001)
losslog = []

for i in range(0, 2000):
    optimizer.zero_grad()

    output = myunet(z)
    loss = criterion(output, target)

    losslog.append(loss.item())

    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        print("Loss: ", losslog[-1])


plt.imsave('final.jpg', output[0].cpu().detach().permute(1, 2, 0).numpy())

# display the loss
plt.figure(figsize=(6, 4))
plt.yscale('log')
plt.plot(losslog, label='loss ({:.4f})'.format(losslog[-1]))
plt.xlabel("Epochs")
plt.legend()
plt.show()
plt.close()
