import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.datasets import DatasetFolder

from Data import Food11,get_BalancedWeights
from models.CNN import CNN
from models.Classifier import Classifier
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

from torch.utils.data.sampler import WeightedRandomSampler

def calc_acc(logits,labels):
  with torch.no_grad():
    acc=(logits.argmax(dim=-1) == labels).float().mean()
  return acc

class Classifier_Trainer():
    def __init__(self,num_class,device='cuda'):
        self.device = device

        # self.model = CNN(output_dim=num_class,k1=256,k2=128).to(device)
        self.model = Classifier().to(device)

    def train(
            self,
            img_ds,
            valid_ds=None,
            batch_size=8, num_epochs=10, num_workers=8,
            ds_balanced=True,
            lr=1e-4,
    ):
        
        if ds_balanced:
          train_w = get_BalancedWeights(img_ds)
          sampler = WeightedRandomSampler(train_w,num_samples=len(train_w),replacement=True)
          img_dl = DataLoader(img_ds, batch_size=batch_size,sampler=sampler,num_workers=num_workers, pin_memory=True)
        else:
          img_dl = DataLoader(img_ds, batch_size=batch_size,shuffle=True,num_workers=num_workers, pin_memory=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        train_acc,valid_acc=0,0
        total_steps = 0
        for epoch in range(num_epochs):
            for i, (images,labels) in enumerate(img_dl, start=1):
                self.model.train()
                x,y = images.to(self.device),labels.to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ----------------------------provide a loss feedback--------------------------- #
                train_acc = calc_acc(logits,y)
                if total_steps%400==0 and valid_ds is not None:
                  print('evaluate valid dataset:')
                  valid_acc=self.evaluation(valid_ds)
                print(f"epoch:{epoch} batch:{i}  loss: {loss.item():f} train_acc: {train_acc:f} valid_acc: {valid_acc:f}")
                total_steps += 1

    def evaluation(self,ds,batch_size=16):
      self.model.eval()
      dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
      p_list,y_list=[],[]

      with torch.no_grad():
          for images, labels in dl:
            x,y = images.to(self.device),labels.to(self.device)
            logits = self.model(x)
            p_list.append(logits)
            y_list.append(y)

      acc=calc_acc(torch.cat(p_list),torch.cat(y_list))
      return acc


    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

if __name__ == '__main__':
    ds_root=r'E:\temp\food-11'
    img_size=128
    batch_size=128
    num_workers=4
    device='cuda'

    food11=Food11(ds_root,(img_size,img_size))
    train_set,valid_set,unlabeled_set,test_set=food11.getDataset()
    
    classifier_trianer=Classifier_Trainer(num_class=11,device=device)
    classifier_trianer.train(train_set,valid_set,batch_size=batch_size,num_epochs=100,num_workers=num_workers)