import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd


class MF(nn.Module):
    def __init__(self, n_coeffs, n_comps, n_pix):
        super(MF, self).__init__()
        self.cfs = nn.Parameter(torch.rand(n_coeffs, n_comps, requires_grad=True))
        self.cmps = nn.Parameter(torch.rand(n_comps, n_pix, requires_grad=True))

    def forward(self):
        return torch.matmul(self.cfs,self.cmps)


def unit_norm(cmps):
    return torch.sum(torch.abs(torch.norm(cmps, dim=1) - (torch.ones(12)*20).to(device)))

ncomps = 12
λ = 1e-6
μ = 1e-7

for j in range(18):
    for i in range(25):
        ds = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_clean.nc")
        ti_nan = (np.count_nonzero(np.isnan(ds.nbart_blue.values), axis=(1,2)))<.66*160000
        ds = ds.isel(time=ti_nan)

        stack = np.empty((0,400,400))
        for fname in ds:
            band = ds[fname]#.values/1e4
            band = band.interpolate_na(dim='time')
            band = band.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')
            stack = np.append(stack, band.values, axis=0)


        stack = stack.reshape(-1, 160000) / 10000
        stack_mean = np.mean(stack, axis=0)
        target = torch.from_numpy(stack-stack_mean).float().to(device)

        ncoeffs = stack.shape[0]
        npix = 160000

        net = MF(ncoeffs, ncomps, npix)
        net.to(device)

        mse = nn.MSELoss(reduction='mean')
            
        opt = optim.AdamW(net.parameters(), lr=1.0)

        n_epoch  = 1000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target) #+ λ*unit_norm(net.cmps)# + μ*torch.norm(net.cfs, p=1)

            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            
            if epoch % 100 == 0:
                print(epoch, torch.norm(net.cfs, p=1).item(), mse(yhat, target).item(), unit_norm(net.cmps).item())

        with torch.no_grad():
            net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
            net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20
            
              
        opt = optim.AdamW(net.parameters(), lr=0.001)

        n_epoch  = 1000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target) #+ λ*unit_norm(net.cmps)# + μ*torch.norm(net.cfs, p=1)
            
            if epoch == 0:
                print(loss.item(), unit_norm(net.cmps).item())

            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            
            if epoch % 100 == 0:
                print(epoch, torch.norm(net.cfs, p=1).item(), mse(yhat, target).item(), unit_norm(net.cmps).item())

        with torch.no_grad():
            net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
            net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20
            
        plt.imshow(net.cfs.detach().cpu().numpy().T, vmin=-1.0, vmax=1.0, aspect=10, cmap='bwr')
        plt.colorbar()
        plt.savefig(f"coeffs_{26*j+i:03d}_dense.png")
        plt.clf()

        opt = optim.AdamW(net.parameters(), lr=0.01)
        n_epoch  = 2000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target) + λ*unit_norm(net.cmps) + μ*torch.norm(net.cfs, p=1)
            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            
            if epoch % 100 == 0:
                print(epoch, torch.norm(net.cfs, p=1).item(), mse(yhat, target).item(), unit_norm(net.cmps).item())

        plt.imshow(net.cfs.detach().cpu().numpy().T, vmin=-1.0, vmax=1.0, aspect=10, cmap='bwr')
        plt.colorbar()
        plt.savefig(f"coeffs_{26*j+i:03d}_sparse.png")
        plt.clf()


