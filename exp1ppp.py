import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import remove_small_objects

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MF(nn.Module):
    def __init__(self, n_coeffs, n_comps, n_pix):
        super(MF, self).__init__()
        self.cfs = nn.Parameter(torch.rand(n_coeffs, n_comps, requires_grad=True))
        self.cmps = nn.Parameter(torch.rand(n_comps, n_pix, requires_grad=True))

    def forward(self):
        return torch.matmul(self.cfs,self.cmps)


def nan_mse_loss(output, target):
    loss = torch.mean((output[target == target] - target[target == target])**2)
    return loss



def generate_patches(patch_hsize, mask):
    i = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))
    j = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))
    t = int(np.random.randint(mask.shape[0], size=1))

    while np.count_nonzero(~mask[t, j-patch_hsize:j+patch_hsize, i-patch_hsize:i+patch_hsize]) != (2*patch_hsize)**2:
        t = int(np.random.randint(mask.shape[0], size=1))
        i = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))
        j = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))

    return t, i, j

print("i,j,band_name,pca_int_mse,pca_orig_mse,pca_patches_mse,nn_int_mse,nn_orig_mse,nn_patches_mse")
for j in range(18):
    for i in range(25):
        ds = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_clean.nc")
        ds = ds.isel(time=(np.count_nonzero(~np.isnan(ds.nbart_blue.values), axis=(1,2)))>160000*.66)
        """
        #ds = xr.open_dataset("Murrumbidgee_near_Bundure__MUR_B3.nc")
        #ds = ds.isel(x=slice(400,800), y=slice(0,400))
        ds = xr.open_dataset(f"/data/pca_act/000_clean.nc")

        ids = ds.where((ds.nbart_blue - ds.nbart_blue.quantile(0.25, dim='time'))<1000)
        ids = ids.isel(time=(np.count_nonzero(~np.isnan(ids.nbart_blue.values), axis=(1,2)))>400*400*.66)
        ids = ids.rolling(time=7, min_periods=3, center=True).median()
        ids = ids.reindex({"time": ds.time})
        ids = ids.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

        mask = (ds.nbart_blue - ids.nbart_blue) > 100 + ids.nbart_blue/2
        mask += (ds.nbart_nir_2 - ids.nbart_nir_2) < -600 + ids.nbart_nir_2/16
        mask = mask.values

        for t in range(mask.shape[0]):
            mask[t] = remove_small_objects(mask[t], 9)
            mask[t] = dilation(mask[t], disk(9))

        ti_nan = (np.count_nonzero(mask, axis=(1,2)))<.66*160000
        ds = ds.where(~mask).isel(time=ti_nan)

        mask = mask[ti_nan]
        """

        patch_hsize = 40
        mask = ds.nbart_blue.isnull().values
        pmask = np.zeros(mask.shape, dtype=np.bool)

        for _ in range(20):
            t, ii, jj = generate_patches(patch_hsize, mask)
            pmask[t, jj-patch_hsize:jj+patch_hsize, ii-patch_hsize:ii+patch_hsize] = True


        for band_name in ["nbart_red","nbart_green","nbart_blue",
                          "nbart_nir_1","nbart_nir_2", "nbart_swir_2","nbart_swir_3"]:
            results = []

            stack = ds[band_name].astype(np.float32) #/ 1e4

            pstack = stack.where(~pmask)
            istack = pstack.interpolate_na(dim='time')
            istack = istack.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

            stack = stack.values.reshape(stack.shape[0], -1)
            pstack = pstack.values.reshape(stack.shape[0], -1)
            istack = istack.values.reshape(stack.shape[0], -1)

            refstack = np.copy(stack)

            stack[~pmask.reshape(-1,160000)] = np.nan

            # PCA
            pca = PCA(n_components=12).fit(istack)
            coeffs = pca.transform(istack)
            pca_decomp = pca.inverse_transform(coeffs)
            results.append(np.mean(np.square(pca_decomp-istack)))
            results.append(np.nanmean(np.square(pca_decomp-refstack)))

            pca_decomp[~pmask.reshape(-1,160000)]=np.nan

            results.append(np.nanmean(np.square(pca_decomp-stack)))
            #print("PCA control gaps", np.nanmean(np.square(pca_decomp-stack)))

            """
            # NN interpolated
            ncomps = 12
            ncoeffs = stack.shape[0]
            npix = 160000

            net = MF(ncoeffs, ncomps, npix)
            net.to(device)
            tmean = np.nanmean(istack, axis=0)
            target = torch.from_numpy(istack-tmean).float().to(device)

            opt = optim.AdamW(net.parameters(), lr=1.0)
            n_epoch  = 1000
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20

            opt = optim.AdamW(net.parameters(), lr=0.001)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20


            rec2 = net().cpu().detach().numpy()+tmean
            results.append(np.nanmean(np.square(rec2-istack)))
            """
            results.append(0)

            # NN missing data
            ncomps = 12
            ncoeffs = stack.shape[0]
            npix = 160000

            net = MF(ncoeffs, ncomps, npix)
            net.to(device)
            tmean = np.nanmean(pstack, axis=0)
            target = torch.from_numpy(pstack-tmean).float().to(device)

            opt = optim.AdamW(net.parameters(), lr=1.0)
            n_epoch  = 1000
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20

            opt = optim.AdamW(net.parameters(), lr=0.001)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20


            rec2 = net().cpu().detach().numpy()+tmean
            results.append(np.nanmean(np.square(rec2-refstack)))

            rec2[~pmask.reshape(-1,160000)]=np.nan
            results.append(np.nanmean(np.square(rec2-stack)))

            print(f"{i},{j},{band_name},{results[0]},{results[1]},{results[2]},{results[3]},{results[4]},{results[5]}")
