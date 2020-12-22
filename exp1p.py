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


ncomps = 12
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

        print(pca_decomp.shape, stack.shape)
        print(np.mean(np.square(pca_decomp - stack)))

        
        stack_mean = np.mean(stack, axis=0)
        target = torch.from_numpy(stack-stack_mean).float().to(device)
        print("AAAA", torch.mean(target), torch.std(target))

        ncoeffs = stack.shape[0]
        npix = 160000

        net = MF(ncoeffs, ncomps, npix)
        net.to(device)

        mse = nn.MSELoss(reduction='mean')

        opt = optim.AdamW(net.parameters(), lr=1.0)
        n_epoch  = 1000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target)
            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())

        opt = optim.AdamW(net.parameters(), lr=0.1)
        n_epoch  = 1000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target)
            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())

        opt = optim.AdamW(net.parameters(), lr=0.01)
        n_epoch  = 1000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target)
            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())

        opt = optim.AdamW(net.parameters(), lr=0.001)
        n_epoch  = 1000
        for epoch in range(n_epoch):
            yhat = net()
            loss = mse(yhat, target)
            net.zero_grad() # need to clear the old gradients
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())



"""
        input = torch.ones(1, device=device)
        tmean = np.nanmean(stack, axis=0)

        target = torch.from_numpy(stack-tmean).float().to(device)
        ncoeffs = stack.shape[0]
        net = ae(ncomps, ncoeffs)
        net.to(device)
        
        optimizer = optim.adamw(net.parameters(), lr=1.0)
        #optimizer = optim.sgd(net.parameters(), lr=0.5)
        epochs = 500
        # training loop:
        for it in range(epochs):
            output = net(input)

            loss = nan_mse_loss(output, target)# + sparsity

            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step()    # does the update

            prev_loss = loss.item()

            if it % 100 == 0:
                print(it, loss.item(), nan_mse_loss(output, target).item())

        optimizer = optim.adamw(net.parameters(), lr=0.01)
        epochs = 1000
        # training loop:
        for it in range(epochs):
            output = net(input)

            loss = nan_mse_loss(output, target)# + sparsity

            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step()    # does the update

            prev_loss = loss.item()

            if it % 100 == 0:
                print(it, loss.item(), nan_mse_loss(output, target).item())
       
        for it in range(100):
            output = net(input)
            loss = nan_mse_loss(output, target)
            loss.backward()
            g_base = net.comps.weight.grad
            with torch.no_grad():
                net.comps.weight -= g_base
            
            print(it, torch.mean(g_base**2))
            
            output = net(input)
            loss = nan_mse_loss(output, target)
            loss.backward()
            g_coeffs = net.coeffs.weight.grad
            with torch.no_grad():
                net.coeffs.weight -= g_coeffs
            
            print(it, loss.item())

        output = net(input)
        output = output.cpu().detach().numpy() + tmean
        
        print(output.shape, stack.shape)
        print(np.mean(np.square(output - stack)))
        exit(0)
ds = xr.open_dataset("/data/pca_act/000_clean.nc")
ti_nan = (np.count_nonzero(np.isnan(ds.nbart_blue.values), axis=(1,2)))<.66*160000
ds = ds.isel(time=ti_nan)

stack = np.empty((0,400,400))
for fname in ds:
    band = ds[fname]#.values/1e4
    band = band.interpolate_na(dim='time')
    band = band.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')
    stack = np.append(stack, band.values, axis=0)


stack = stack.reshape(-1, 160000) / 10000
print(stack.max(), stack.min())

np.save("stack", stack)


exit(0)

class ae(nn.module):
    def __init__(self, n_comps, n_coeffs):
        super(ae, self).__init__()
        self.n_coeffs = n_coeffs
        self.n_comps = n_comps
        self.comps = nn.linear(1, self.n_comps*400*400, bias=false)
        self.coeffs = nn.linear(1, self.n_comps*self.n_coeffs, bias=false)

    def forward(self, x):
        base = self.comps(x).reshape(self.n_comps, 400*400)
        coeffs = self.coeffs(x).view(self.n_coeffs, self.n_comps)
        return torch.matmul(coeffs, base)
        return torch.einsum('ki,kj->ji', conv1.view(self.n_comps,400*400), coeffs)

def nan_mse_loss(output, target):
    loss = torch.mean((output[target == target] - target[target == target])**2)
    return loss

stack = np.load("stack.npy")
pca = pca(n_components=ncomps).fit(stack)
coeffs = pca.transform(stack)
pca_decomp = pca.inverse_transform(coeffs)

print(pca_decomp.shape, stack.shape)
print(np.mean(np.square(pca_decomp - stack)))


input = torch.ones(1, device=device)
net = ae(ncomps, ncoeffs)
net.to(device)

mse = nn.mseloss(reduction='mean')
optimizer = optim.adamw(net.parameters(), lr=1.0)
epochs = 500
# training loop:
for it in range(epochs):
    output = net(input)

    loss = mse(output, target)# + sparsity
    #loss = nan_mse_loss(output, target)# + sparsity

    optimizer.zero_grad()   # zero the gradient buffers
    loss.backward()
    optimizer.step()    # does the update

    prev_loss = loss.item()

    if it % 100 == 0:
        print(it, loss.item())

exit(0)



class ae(nn.module):
    def __init__(self, n_comps, n_coeffs):
        super(ae, self).__init__()
        self.n_coeffs = n_coeffs
        self.n_comps = n_comps
        self.comps = nn.linear(1, self.n_comps*400*400, bias=false)
        self.coeffs = nn.linear(1, self.n_comps*self.n_coeffs, bias=false)

    def forward(self, x):
        base = self.comps(x).reshape(self.n_comps, 400*400)
        coeffs = self.coeffs(x).view(self.n_coeffs, self.n_comps)
        return torch.matmul(coeffs, base)
        return torch.einsum('ki,kj->ji', conv1.view(self.n_comps,400*400), coeffs)

def nan_mse_loss(output, target):
    loss = torch.mean((output[target == target] - target[target == target])**2)
    return loss

ncomps = 12
"""
