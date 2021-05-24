import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,output_dim=2,k1=64,k2=256):
        super(CNN, self).__init__()
        def ConvBlock(c1, c2):
            block = nn.Sequential(
                nn.Conv2d(c1, c2, 3, 1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            return block

        def FcBlock(n1, n2):
            block = nn.Sequential(
                nn.Linear(n1, n2),
                nn.BatchNorm1d(n2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False)
            )
            return block
        self.Conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((256, 256)),
            *ConvBlock(3, k1),
            *ConvBlock(k1, k1),
            *ConvBlock(k1, k1),
            *ConvBlock(k1, k1),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.Fc = nn.Sequential(
            *FcBlock(k1*8*8, k2),
            *FcBlock(k2, k2),
            *FcBlock(k2, k2),
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(k2, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        out = self.Conv(img)
        out = out.flatten(start_dim=1)
        out = self.Fc(out)
        validity = self.adv_layer(out)
        return validity

if __name__ == '__main__':
    model=CNN(output_dim=11)
    images=torch.ones([4,3,512,512])
    p=model(images)
    print(p.shape)