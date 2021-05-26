import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class Vgg16(nn.Module):
    def __init__(self,output_dim=2, pretrained=True):
        super().__init__()
        # self.model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=pretrained)
        self.model = models.vgg16(pretrained)
        self.ac_layer=nn.Sequential(
            nn.Linear(1000, output_dim)
        )

    def forward(self, img):
        out = self.model(img)
        out = self.ac_layer(out)
        return out

def get_boneneck(name='vgg16',pretrained=True):
    if name=='vgg16':
        pass
    else:
        print(torch.hub.list('pytorch/vision'))
        raise NotImplementedError('Not yet implemention!')
    return None

if __name__ == '__main__':
    model=Vgg16(output_dim=11)
    print(model)