import xarray as xr
import numpy as np
from scipy import signal
from skimage import morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)
print(device)


class Net(nn.Module):

    def __init__(self, n_coeffs):
        super(Net, self).__init__()
        self.n_coeffs = n_coeffs
        self.fc1 = nn.Linear(1, self.n_coeffs, bias=False)  # 6*6 from image dimension
        self.fc2 = nn.Linear(1, 160000, bias=False)

    def forward(self, x):
        coeffs = self.fc1(x)
        base = self.fc2(x)
        return torch.mul(coeffs.view(self.n_coeffs, 1), base.view(1, 160000))


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

    # 1.- Create mask for reflectances with deviations more than 0.07 from lower quartile
    qmask = ((blue - blue.quantile(0.25, dim='time'))>.07).values

    # 2.- Remove small objects (< 36 pixels)
    for i in range(qmask.shape[0]):
        qmask[i] = morphology.remove_small_objects(qmask[i], 36)
        
    blue = blue.where(~qmask)

    # 3.- Grow a 9x9 buffer around NaN pixels 
    blue = buffer_nans(blue, 9)

    # 4.- Discard frames with more than 33% missing pixels
    blue = blue.isel(time=(np.count_nonzero(np.isnan(blue.values), axis=(1,2))/(400*400))<.33)
    
    return blue


for j in range(10,18):
    for i in range(26):
        ds2018 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2018.nc")
        ds2019 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2019.nc")

        ds = xr.concat([ds2018, ds2019], dim='time').sortby('time')
        print(ds.time.shape)

        # 1. Create blue mask
        blue = generate_blue_mask(ds)
        print(blue.time)

        # 2. Apply blue mask over entire dataset
        ds = ds.sel(time=blue.time).where(~np.isnan(blue))

        # 3. Apply blue mask over entire dataset
        stack = np.empty((0,400,400))
        for vname in ds:
            stack = np.append(stack, ds[vname].values/1e4, axis=0)

        stack = stack.reshape(stack.shape[0], -1)
        ncoeffs = stack.shape[0]

        input = torch.ones(1, device=device)
        criterion = nan_mse_loss

        target = torch.from_numpy(stack-np.mean(stack, axis=0)).float().to(device)

        for i in range(12):
            print(i)
            net = Net(ncoeffs)
            net.to(device)
            optimizer = optim.Adam(net.parameters(), lr=0.001)

            for j in range(10000):
                # training loop:
                optimizer.zero_grad()   # zero the gradient buffers
                output = net(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()    # Does the update

                if j % 1000 == 0:
                    loss = criterion(output, target)
                    print(j, loss)

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

            np.save(f"q_coeffs{i:02d}", coeffs)
            np.save(f"q_offscale{i:02d}", np.array([offset,scale]))
            np.save(f"q_base{i:02d}", qbase)

            residual = target.cpu().detach().numpy() - coeffs*rbase.reshape(1,-1)
            target = torch.from_numpy(residual).to(device)
