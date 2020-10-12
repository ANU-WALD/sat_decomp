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

#(i,j)
#(6,10) -> Canberra Civic
#(12,4) -> Reflective terrain

#for j in range(1,18):
for _ in [1]:
    #for i in range(25):
    for i, j in [(6,10), (12,4)]:
        print(i,j)

        ds2018 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2018.nc")
        ds2019 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2019.nc")

        ds = xr.concat([ds2018, ds2019], dim='time').sortby('time')

        # 1. Create blue mask
        blue = generate_blue_mask(ds)
        np.save(f"{j:02d}_{i:02d}_times", blue.time.values)

        # 2. Apply blue mask over entire dataset
        ds = ds.sel(time=blue.time).where(~np.isnan(blue))

        # 3. Apply blue mask over entire dataset
        stack = np.empty((0,400,400))

        for vname in ds:
            stack = np.append(stack, ds[vname].values/1e4, axis=0)
        
        stack = stack.reshape(stack.shape[0], -1)
        ncoeffs = stack.shape[0]

        input = torch.ones(1, device=device)
        #input = torch.ones(1, device='cpu')
        tmean = np.nanmean(stack, axis=0)
        np.save(f"{j:02d}_{i:02d}_mean", tmean)
        target = torch.from_numpy(stack-tmean).float().to(device)
        #target = torch.from_numpy(stack-tmean).float()

        for pc_i in range(6):
            net = Net(ncoeffs)
            net.to(device)
            optimizer = optim.Adam(net.parameters(), lr=0.1)

            prev_loss = 10.0
            patience = 0
            for it in range(10000):
                # training loop:
                output = net(input)
                loss = nan_mse_loss(output, target)

                # Patience 
                if (prev_loss-loss.item()) < 1e-7:
                    patience += 1
                else:
                    patience = 0

                if patience == 10:
                    break

                optimizer.zero_grad()   # zero the gradient buffers
                loss.backward()
                optimizer.step()    # Does the update

                prev_loss = loss.item()

            params = list(net.parameters())
            coeffs = params[0].cpu().detach().numpy()
            base = params[1].cpu().detach().numpy()
            print("Range:", base.max()-base.min(), "QRes:", (base.max()-base.min())/255)

            offset = base.min()
            scale = 1/(base.max()-base.min())
            nbase = (base-offset)*scale
            print("NRange:", nbase.max(), nbase.min())

            qbase = np.rint(nbase*255).astype(np.uint8)

            rbase = offset + ((qbase.astype(np.float32)/255)/scale)
            print("QError:", np.mean(np.square(rbase-base)))

            np.save(f"{j:02d}_{i:02d}_q_coeffs{pc_i:02d}", coeffs)
            np.save(f"{j:02d}_{i:02d}_q_offscale{pc_i:02d}", np.array([offset,scale]))
            np.save(f"{j:02d}_{i:02d}_q_base{pc_i:02d}", qbase)

            residual = target.cpu().detach().numpy() - coeffs*rbase.reshape(1,-1)
            target = torch.from_numpy(residual).to(device)
