import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class AE(nn.Module):
    def __init__(self, n_comps, n_coeffs):
        super(AE, self).__init__()
        self.n_coeffs = n_coeffs
        self.n_comps = n_comps
        self.D = nn.Linear(1, self.n_comps*404*404, bias=False)
        self.conv1 = nn.Conv2d(self.n_comps, 2*self.n_comps, kernel_size=5, bias=False)
        self.coeffs = nn.Linear(1, 2*self.n_comps*self.n_coeffs, bias=False)

    def forward(self, x):
        base = self.D(x)
        conv1 =  torch.tanh(self.conv1(base.reshape(1, self.n_comps, 404, 404)))
        coeffs = self.coeffs(x).view(2*self.n_comps, self.n_coeffs)
        return torch.einsum('ki,kj->ji', conv1.view(2*self.n_comps,400*400), coeffs)


def nan_mse_loss(output, target):
    loss = torch.mean((output[target == target] - target[target == target])**2)
    return loss


for j in range(18):
    for i in range(25):
        ds = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_clean.nc")
        ti_nan = (np.count_nonzero(np.isnan(ds.nbart_blue.values), axis=(1,2)))<.66*160000
        ds = ds.isel(time=ti_nan)
        np.save(f"{j:02d}_{i:02d}_times", ds.time.values)

        stack = np.empty((0,400,400))
        for fname in ds:
            band = ds[fname].values/1e4
            stack = np.append(stack, band, axis=0)

        stack = stack.reshape(stack.shape[0], -1)

        ncomps = 12
        ncoeffs = stack.shape[0]

        input = torch.ones(1, device=device)
        tmean = np.nanmean(stack, axis=0)
        np.save(f"{j:02d}_{i:02d}_mean", tmean.astype(np.float32)

        target = torch.from_numpy(stack-tmean).float().to(device)

        net = AE(ncomps, ncoeffs)
        net.to(device)
        optimizer = optim.AdamW(net.parameters(), lr=0.1)

        epochs = 1500
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

        torch.save(net, f"{j:02d}_{i:02d}_net.pt")

        params = list(net.parameters())
        base = params[0].cpu().detach().numpy()
        kernel = params[1].cpu().detach().numpy()
        coeffs = params[2].cpu().detach().numpy()

        np.save(f"{j:02d}_{i:02d}_base", base)
        np.save(f"{j:02d}_{i:02d}_kernel", kernel)
        np.save(f"{j:02d}_{i:02d}_coeffs", coeffs)
