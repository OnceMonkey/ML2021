from models.Classifier import Classifier
from models.CNN import CNN
from models.Public import Vgg16

import torchvision


def get_model(name,num_class=11,pretrained=False):
    if name=='vgg16':
        return Vgg16(num_class,pretrained)
    elif name=='tiny_cnn':
        return Classifier()
    elif name=='cnn':
        return CNN(num_class)
    elif name == 'resnet18':
        return torchvision.models.resnet18(num_classes=num_class,pretrained=pretrained)
    else:
        raise NotImplementedError('not yet implementation!')


