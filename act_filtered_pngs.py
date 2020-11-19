import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import os.path

from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import remove_small_objects

for j in range(18):
    for i in range(25):
        print(i, j, f"/data/pca_act/{26*j+i:03d}_2018.nc")

        if os.path.isfile(f"/data/pca_act/{26*j+i:03d}.png"):
            continue

        ds2018 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2018.nc")
        ds2019 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2019.nc")
        ds = xr.concat([ds2018, ds2019], dim='time').sortby('time')
        
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

        p = ds.where(~mask)[['nbart_red','nbart_green','nbart_blue']].clip(0,2000).to_array().plot.imshow(col='time', col_wrap=6)
        print(f"/data/pca_act/{26*j+i:03d}.png")
        p.fig.savefig(f"/data/pca_act/{26*j+i:03d}.png")
        plt.close(p.fig)
        ds.where(~mask).to_netcdf(f"/data/pca_act/{26*j+i:03d}_clean.nc")
