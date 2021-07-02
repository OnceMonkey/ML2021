
import torch


a=torch.Tensor([[1,2,3],[3,2,1]])
print(torch.max(a,dim=1))
