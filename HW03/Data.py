import os
from torchvision.datasets import DatasetFolder,ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

def get_BalancedWeights(ds:DatasetFolder):
    datas=pd.DataFrame(ds.samples,columns=['paths','label'])
    distribution=pd.value_counts(datas['label'])
    weights=[1/float(distribution[i]) for i in datas['label']]
    # ----------------------------展示数据分布--------------------------- #
    print(datas['label'],distribution,weights)
    import matplotlib.pyplot as plt
    datas['label'].hist()
    plt.show()
    return weights

class Food11():
    def __init__(self, ds_root=None,img_size=(128, 128)):
        self.ds_root = ds_root
        self.train_tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.test_tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    

    def getDataset(self):
        train_path=os.path.join(self.ds_root,'training/labeled')
        valid_path=os.path.join(self.ds_root,'validation')
        unlabeled_path=os.path.join(self.ds_root,'training/unlabeled')
        test_path=os.path.join(self.ds_root,'testing')

        train_set = ImageFolder(train_path,transform=self.train_tfm)
        valid_set = ImageFolder(valid_path,transform=self.test_tfm)
        unlabeled_set = ImageFolder(unlabeled_path, transform=self.train_tfm)
        test_set = ImageFolder(test_path, transform=self.test_tfm)

        return train_set, valid_set,unlabeled_set,test_set

        

if __name__ == '__main__':
    img_size=128
    ds_root=r'E:\temp\food-11'
    food11=Food11(ds_root,(img_size,img_size))
    train_set,valid_set,unlabeled_set,test_set=food11.getDataset()
    # for image,label in unlabeled_set:
    #     print(image.shape,label)
    
    from torch.utils.data.sampler import WeightedRandomSampler
    from torch.utils.data import DataLoader
    train_w = get_BalancedWeights(train_set)
    sampler = WeightedRandomSampler(train_w,num_samples=len(train_w),replacement=True)
    print(len(sampler),len(train_w))
    img_dl = DataLoader(train_set, batch_size=128,sampler=sampler,num_workers=4, pin_memory=True)
    img_dl = DataLoader(train_set, batch_size=128,shuffle=True,num_workers=4, pin_memory=True)
    # for i, (images,labels) in enumerate(img_dl, start=1): 
        # print(images.shape,labels.shape)
    
    