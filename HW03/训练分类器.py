import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler




def calc_acc(logits,labels):
  with torch.no_grad():
    acc=(logits.argmax(dim=-1) == labels).float().mean()
  return acc

class Classifier_Trainer():
    def __init__(self,model,device='cuda'):
        self.device = device

        self.model = model.to(device)

    def get_pseudo_labels(self,unlabeled_ds,threshold=0.65,num_workers=4):
      print(f'开始半监督增强数据! 总数据量:{len(unlabeled_ds)}')
      self.model.eval()
      data_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=False,num_workers=num_workers)
      aug_images,aug_labels=[],[]
      for images,_ in data_loader:
          x = images.to(self.device)
          with torch.no_grad():
              logits = self.model(x)
          probs = F.softmax(logits,dim=-1)
          info=torch.max(probs,dim=-1)
          for i,(p,label) in enumerate(zip(info[0],info[1])):
            if p>threshold:
              aug_images.append(images[i])
              aug_labels.append(label)
      if len(aug_images)>0:
        aug_images,aug_labels=torch.stack(aug_images),torch.stack(aug_labels)
        semi_ds=TensorDataset(aug_images,aug_labels)
        print(f'半监督学习数据增强, 阈值:{threshold}, 加入训练量:{len(semi_ds)}')
      else:
        semi_ds=None
        print(f'半监督学习数据增强, 阈值:{threshold}, 加入训练量:{0}')
      return semi_ds

    def train(
            self,
            img_ds,
            valid_ds=None,
            batch_size=8, num_epochs=10, num_workers=8,
            lr=3e-4,
            unlabeled_ds=None, threshold=0.65,
            ds_balanced=False,
            cpt_frequency=400,
            valid_frequency=400,

    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        train_acc,valid_acc=0,0
        total_steps = 0
        for epoch in range(num_epochs):
          # ----------------------------构造数据加载过程--------------------------- #
          if ds_balanced:
            if unlabeled_ds is None:
              train_ds=img_ds
            else:
              pseudo_set=self.get_pseudo_labels(unlabeled_ds,threshold)
              train_ds=ConcatDataset([img_ds, pseudo_set]) if pseudo_set is not None else img_ds
            train_w = get_BalancedWeights(train_ds)
            sampler = WeightedRandomSampler(train_w,num_samples=len(train_w),replacement=True)
            img_dl = DataLoader(img_ds, batch_size=batch_size,sampler=sampler,num_workers=num_workers, pin_memory=True)
          else:
            if unlabeled_ds is None:
              train_ds=img_ds
            else:
              pseudo_set=self.get_pseudo_labels(unlabeled_ds,threshold)
              train_ds=ConcatDataset([img_ds, pseudo_set]) if pseudo_set is not None else img_ds
            img_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True,num_workers=num_workers, pin_memory=True)
          # ----------------------------开始训练--------------------------- #
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
              if total_steps%valid_frequency==0 and valid_ds is not None:
                print('evaluate valid dataset:')
                valid_acc=self.evaluation(valid_ds)
              if (total_steps+1)%cpt_frequency==0:
                self.save('./cps/model.pth')
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
    import argparse
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--model_name',type=str,default='tiny_cnn')
    parser.add_argument('--img_size',type=int,default=128)
    parser.add_argument('--num_class',type=int,default=11)
    parser.add_argument('--pretrained',type=bool,default=False)

    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--num_epochs',type=int,default=500)
    parser.add_argument('--num_workers',type=int,default=4)

    parser.add_argument('--lr', type=float, default=3e-4)

    parser.add_argument('--valid_frequency', type=int, default=400)
    parser.add_argument('--cpt_frequency', type=int, default=400)

    parser.add_argument('--ds_balanced', type=bool, default=False)
    parser.add_argument('--use_unlabeled', type=bool, default=False)
    parser.add_argument('--threshold', type=int, default=0.65)

    parser.add_argument('--device', type=str, default='cuda')

    args=parser.parse_args()
    print(args)

    from Data import Food11, get_BalancedWeights

    ds_root = r'E:\temp\food-11'
    food11=Food11(ds_root,(args.img_size,args.img_size))
    train_set,valid_set,unlabeled_set,test_set=food11.getDataset()
    if not args.use_unlabeled:
        unlabeled_set=None

    from models.get_model import get_model
    model=get_model(args.model_name,args.num_class,args.pretrained)


    classifier_trianer=Classifier_Trainer(model,device=args.device)
    classifier_trianer.train(
        train_set, valid_set,
        batch_size=args.batch_size, num_epochs=args.num_epochs, num_workers=args.num_workers,
        lr=args.lr,
        unlabeled_ds=unlabeled_set, threshold=args.threshold,
        ds_balanced=args.ds_balanced,
        valid_frequency=args.valid_frequency,
        cpt_frequency=args.cpt_frequency,
    )