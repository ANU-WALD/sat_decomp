import xarray as xr
import numpy as np
from scipy import signal
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import remove_small_objects

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Net(nn.Module):

    def __init__(self, n_coeffs):
        super(Net, self).__init__()
        self.n_coeffs = n_coeffs
        self.fc1 = nn.Linear(1, self.n_coeffs, bias=False)
        self.fc2 = nn.Linear(1, 160000, bias=False)

    def forward(self, x):
        coeffs = self.fc1(x)
        base = self.fc2(x)
        return torch.matmul(coeffs.unsqueeze(1), base.unsqueeze(0))


def nan_mse_loss(output, target):
    loss = torch.mean((output[target == target] - target[target == target])**2)
    return loss


def buffer_nans(da, kn):
    k = np.zeros((kn,kn))
    k[kn//2,kn//2] = 1

    arr = da.values
    mask = np.ones(arr.shape).astype(np.float32)

    for i in range(arr.shape[0]):
        mask[i,:,:] = signal.convolve2d(arr[i,:,:], k, boundary='fill', mode='same')

    return da.where(~np.isnan(mask))

def generate_blue_mask(ds):
    blue = ds.nbart_blue.astype(np.float32) / 1e4

    # 1.- Remove images with less than 1000 valid pixels
    blue = blue.isel(time=(np.count_nonzero(~np.isnan(blue.values), axis=(1,2)))>1000)

    # 2.- Create mask for reflectances with deviations more than 10% from lower quartile
    qmask = ((blue - blue.quantile(0.25, dim='time'))>.1).values

    # 3.- Remove small objects (< 36 pixels) and grow a 15 pixel disk buffer around remaining objects
    for i in range(qmask.shape[0]):
        qmask[i] = remove_small_objects(qmask[i], 36)
        qmask[i] = dilation(qmask[i], disk(15))
        qmask[i][np.isnan(blue[i].values)] = False

    # 4.- Apply mask
    blue = blue.where(~qmask)

    # 5.- Discard frames with more than 33% missing pixels (relative to the initial valid pixels)
    blue = blue.isel(time=(np.count_nonzero(qmask, axis=(1,2)) / (1+np.count_nonzero(~np.isnan(blue.values), axis=(1,2))))<.33)

    return blue

ds = xr.open_dataset(f"Murrumbidgee_near_Bundure__MUR_B3.nc")
ds = ds.isel(x=slice(400,800), y=slice(0,400))

# 1. Create blue mask
blue = generate_blue_mask(ds)
np.save(f"baseline_times", blue.time.values)

# 2. Apply blue mask over entire dataset
ds = ds.sel(time=blue.time).where(~np.isnan(blue))

# 3. Apply blue mask over entire dataset
stack = np.empty((0,400,400))

for vname in ds:
    stack = np.append(stack, ds[vname].values/1e4, axis=0)
        
stack = stack.reshape(stack.shape[0], -1)
ncoeffs = stack.shape[0]

input = torch.ones(1, device=device)
tmean = np.nanmean(stack, axis=0)
np.save(f"baseline_mean", tmean)
target = torch.from_numpy(stack-tmean).float().to(device)

for pc_i in range(24):
    print("Component:", pc_i)
    net = Net(ncoeffs)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    prev_loss = 10.0
    patience = 0
    for it in range(10000):
        # training loop:
        output = net(input)
        loss = nan_mse_loss(output, target)

        # patience 
        if (prev_loss-loss.item()) < 1e-7:
            patience += 1
        else:
            patience = 0

        if patience == 10:
            break

        optimizer.zero_grad()   # zero the gradient buffers
        loss.backward()
        optimizer.step()    # do the update

        prev_loss = loss.item()

    params = list(net.parameters())
    coeffs = params[0].cpu().detach().numpy()
    base = params[1].cpu().detach().numpy()

    np.save(f"baseline_base{pc_i:02d}", base)
    np.save(f"baseline_coeffs{pc_i:02d}", coeffs)
            
    residual = target.cpu().detach().numpy() - coeffs*base.reshape(1,-1)
    target = torch.from_numpy(residual).to(device)
