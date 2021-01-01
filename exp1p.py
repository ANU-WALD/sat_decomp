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

        pca = PCA(n_components=ncomps).fit(stack)
        coeffs = pca.transform(stack)
        pca_decomp = pca.inverse_transform(coeffs)

        print("PCA MSE 12 comp", np.mean(np.square(pca_decomp - stack)))
        
        pca = PCA(n_components=11).fit(stack)
        coeffs = pca.transform(stack)
        pca_decomp = pca.inverse_transform(coeffs)

        print("PCA MSE 11 comp", np.mean(np.square(pca_decomp - stack)))

        pca = PCA(n_components=10).fit(stack)
        coeffs = pca.transform(stack)
        pca_decomp = pca.inverse_transform(coeffs)

        print("PCA MSE 10 comp", np.mean(np.square(pca_decomp - stack)))

        pca = PCA(n_components=9).fit(stack)
        coeffs = pca.transform(stack)
        pca_decomp = pca.inverse_transform(coeffs)

        print("PCA MSE 9 comp", np.mean(np.square(pca_decomp - stack)))

        pca = PCA(n_components=8).fit(stack)
        coeffs = pca.transform(stack)
        pca_decomp = pca.inverse_transform(coeffs)

        print("PCA MSE 8 comp", np.mean(np.square(pca_decomp - stack)))

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
            
        cfs = net.cfs.detach().cpu().numpy()
        print("cfs size:", cfs.shape, cfs.size)
        print("Zeros before:", np.sum(np.isclose(cfs, np.zeros(cfs.shape), rtol=1e-03, atol=1e-04)))
        print(f"Zeros after: 0.1->{np.sum(np.abs(cfs)<0.1)} 0.01->{np.sum(np.abs(cfs)<0.01)} 0.001->{np.sum(np.abs(cfs)<0.001)} 0.0001->{np.sum(np.abs(cfs)<0.0001)} 0.00001->{np.sum(np.abs(cfs)<0.00001)} 0.000001->{np.sum(np.abs(cfs)<0.000001)} 0.0000001->{np.sum(np.abs(cfs)<0.0000001)}")

        plt.imshow(cfs.T, vmin=-1.0, vmax=1.0, aspect=10, cmap='bwr')
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

        cfs = net.cfs.detach().cpu().numpy()
        print("Zeros after:", np.sum(np.isclose(cfs, np.zeros(cfs.shape), rtol=1e-03, atol=1e-04)))
        print(f"Zeros after: 0.1->{np.sum(np.abs(cfs)<0.1)} 0.01->{np.sum(np.abs(cfs)<0.01)} 0.001->{np.sum(np.abs(cfs)<0.001)} 0.0001->{np.sum(np.abs(cfs)<0.0001)} 0.00001->{np.sum(np.abs(cfs)<0.00001)} 0.000001->{np.sum(np.abs(cfs)<0.000001)} 0.0000001->{np.sum(np.abs(cfs)<0.0000001)}")
        plt.imshow(cfs.T, vmin=-1.0, vmax=1.0, aspect=10, cmap='bwr')
        plt.colorbar()
        plt.savefig(f"coeffs_{26*j+i:03d}_sparse.png")
        plt.clf()


        opt = optim.AdamW(net.parameters(), lr=0.01)
        n_epoch  = 2000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target) + λ*unit_norm(net.cmps) + 2*μ*torch.norm(net.cfs, p=1)
            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            
            if epoch % 100 == 0:
                print(epoch, torch.norm(net.cfs, p=1).item(), mse(yhat, target).item(), unit_norm(net.cmps).item())

        cfs = net.cfs.detach().cpu().numpy()
        print("Zeros after 2x:", np.sum(np.isclose(cfs, np.zeros(cfs.shape), rtol=1e-03, atol=1e-04)))
        print(f"Zeros after 2x: 0.1->{np.sum(np.abs(cfs)<0.1)} 0.01->{np.sum(np.abs(cfs)<0.01)} 0.001->{np.sum(np.abs(cfs)<0.001)} 0.0001->{np.sum(np.abs(cfs)<0.0001)} 0.00001->{np.sum(np.abs(cfs)<0.00001)} 0.000001->{np.sum(np.abs(cfs)<0.000001)} 0.0000001->{np.sum(np.abs(cfs)<0.0000001)}")
        
        opt = optim.AdamW(net.parameters(), lr=0.01)
        n_epoch  = 2000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target) + λ*unit_norm(net.cmps) + 4*μ*torch.norm(net.cfs, p=1)
            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            
            if epoch % 100 == 0:
                print(epoch, torch.norm(net.cfs, p=1).item(), mse(yhat, target).item(), unit_norm(net.cmps).item())

        cfs = net.cfs.detach().cpu().numpy()
        print("Zeros after 4x:", np.sum(np.isclose(cfs, np.zeros(cfs.shape), rtol=1e-03, atol=1e-04)))
        print(f"Zeros after 4x: 0.1->{np.sum(np.abs(cfs)<0.1)} 0.01->{np.sum(np.abs(cfs)<0.01)} 0.001->{np.sum(np.abs(cfs)<0.001)} 0.0001->{np.sum(np.abs(cfs)<0.0001)} 0.00001->{np.sum(np.abs(cfs)<0.00001)} 0.000001->{np.sum(np.abs(cfs)<0.000001)} 0.0000001->{np.sum(np.abs(cfs)<0.0000001)}")
        plt.imshow(cfs.T, vmin=-1.0, vmax=1.0, aspect=10, cmap='bwr')
        plt.colorbar()
        plt.savefig(f"coeffs_{26*j+i:03d}_sparse4x.png")
        plt.clf()
        exit()
