import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import remove_small_objects

i,j=11,3
ds2018 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2018.nc")
ds2019 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2019.nc")

ds = xr.concat([ds2018, ds2019], dim='time').sortby('time')

fds = ds.where((ds.nbart_blue - ds.nbart_blue.quantile(0.25, dim='time'))<500)
fds = fds.isel(time=(np.count_nonzero(~np.isnan(fds.nbart_blue.values), axis=(1,2)))>400*400*.66)
fds = fds.rolling(time=7, min_periods=3, center=True).median()
fds = fds.reindex({"time": ds.time})

ids = fds.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

#for bias in [-1000,-800,-600,-400,-200,-100,0,100,200,400,600,800,1000]:
for bias in [400,600,800,1000]:
    for slope in [-8,-4,-2,-1,-0.5,-0.25,0.25,0.5,1,2,4,8]:

        if bias > 0 and slope > 0:
            continue
        
        print(bias, slope)

        mask = (ds.nbart_swir_2 - ids.nbart_swir_2) < (bias + ids.nbart_swir_2/slope)
        mask = mask.values

        for i in range(mask.shape[0]):
            mask[i] = remove_small_objects(mask[i], 9)
            mask[i] = dilation(mask[i], disk(9))

        p = ds.where(~mask)[['nbart_red','nbart_green','nbart_blue']].clip(0,500).to_array().plot.imshow(col='time', col_wrap=6)
        p.fig.savefig(f"/data/act_swir_{bias:+05d}_{slope}.png")
        plt.close(p.fig) 
        
        mask = (ds.nbart_nir_2 - ids.nbart_nir_2) < (bias + ids.nbart_nir_2/slope)
        mask = mask.values

        for i in range(mask.shape[0]):
            mask[i] = remove_small_objects(mask[i], 9)
            mask[i] = dilation(mask[i], disk(9))

        p = ds.where(~mask)[['nbart_red','nbart_green','nbart_blue']].clip(0,500).to_array().plot.imshow(col='time', col_wrap=6)
        p.fig.savefig(f"/data/act_nir_{bias:+05d}_{slope}.png")
        plt.close(p.fig) 

