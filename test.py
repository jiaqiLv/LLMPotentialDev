import torch
import torch.nn as nn


if __name__ == "__main__":
    n_rep = 5
    x = torch.zeros(size=(10,20,30,40))
    a,b,c,d=x.shape
    duplicate_x = x[:,:,:,None,:].expand(a,b,c,n_rep,d).reshape(a,b,c*n_rep,d)
    print('duplicate_x.shape:', duplicate_x.shape)