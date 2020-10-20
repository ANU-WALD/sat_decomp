from flask import Flask
from flask import request
from flask import send_file
from flask import render_template
from flask import make_response

from pyproj import Proj, Transformer
import imageio
from matplotlib import cm
import numpy as np
import xarray as xr
import math
#import numexpr as ne

import os
import io


webmerc_proj = 'epsg:3857'
wgs84_proj = 'epsg:4326'
aaea_proj = 'epsg:3577'

s2_act_geot = [1506645.0, 10.0, 0.0, -3932965.0, 0.0, -10.0]

pca_coeffs = "../data/%02d_%02d_pca_coeffs.npy"
pca_pcs = "../data/%02d_%02d_pca_pcs.npy"
pca_mean = "../data/%02d_%02d_pca_mean.npy"

n_bands = 7

tile_size = 400


app = Flask(__name__)

def get_original_tile(i, j, t):
    ds = xr.open_dataset(f"/data/pca_act/{26*i+j:03d}_2018.nc")
    #ds2019 = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_2019.nc")
    #ds = xr.concat([ds2018, ds2019], dim='time').sortby('time')


    r = ds.nbart_red.isel(time=t).values/10000
    #g = ds.nbart_green.isel(time=0).values/10000
    #b = ds.nbart_blue.isel(time=0).values/10000
    nir = ds.nbart_nir_1.isel(time=t).values/10000

    return (nir-r)/(nir+r)
    return np.dstack((r,g,b))

def get_compressed_tile(i, j, t):
    coeffs = np.load(pca_coeffs%(i,j))
    pcs = np.load(pca_pcs%(i,j)).reshape(-1,tile_size*tile_size)
    mean = np.load(pca_mean%(i,j))
    stride = coeffs.shape[0]//n_bands

    r = mean+(coeffs[None,t+0*stride,:].dot(pcs)).reshape(tile_size,tile_size)
    #g = mean+(coeffs[None,t+1*stride,:].dot(pcs)).reshape(tile_size,tile_size)
    #b = mean+(coeffs[None,t+2*stride,:].dot(pcs)).reshape(tile_size,tile_size)
    nir = mean+(coeffs[None,t+3*stride,:].dot(pcs)).reshape(tile_size,tile_size)

    return (nir-r)/(nir+r)
    return np.dstack((r,g,b))


def bbox2tile(bbox, layer, band, t, im_size, proj):
    trans = Transformer.from_crs(proj, aaea_proj)

    x_tl, y_tl = trans.transform(bbox[0], bbox[3])
    x_tr, y_tr = trans.transform(bbox[2], bbox[3])
    x_br, y_br = trans.transform(bbox[2], bbox[1])
    x_bl, y_bl = trans.transform(bbox[0], bbox[1])

    max_x = max(x_tr, x_br)
    min_x = min(x_bl, x_tl)
    max_y = max(y_tl, y_tr)
    min_y = min(y_bl, y_br)

    x0 = int(math.floor((min_x - s2_act_geot[0])/4000))
    x1 = int(math.ceil((max_x - s2_act_geot[0])/4000))
    y0 = int(math.floor((s2_act_geot[3] - max_y)/4000))
    y1 = int(math.ceil((s2_act_geot[3] - min_y)/4000))

    arr = None
    for i in range(x0, x1):
        for j in range(y0, y1):
            a = get_partial_tile(bbox, layer, band, i, j, t, im_size, proj)
            if arr is None:
                arr = a
                continue
                
            arr += a
    
    return arr

def get_partial_tile(bbox, layer, b, i, j, t, im_size=256, proj=wgs84_proj):
    pixel_size = ((bbox[2] - bbox[0]) / im_size, (bbox[3] - bbox[1]) / im_size)

    #arr = np.zeros((im_size,im_size,3), dtype=np.float32)
    
    lons = []
    lats = []
    for lat in np.arange(bbox[3], bbox[1], -pixel_size[1]):
        for lon in np.arange(bbox[0], bbox[2], pixel_size[0]):
            lons.append(lon)
            lats.append(lat)
    
    trans = Transformer.from_crs(proj, aaea_proj)
    xs, ys = trans.transform(lons, lats)
    
    origin = (1506645.0+i*4000, -3932965.0-j*4000)
    arr = np.zeros((im_size,im_size), dtype=np.float32)

    if layer == 'orig':
        tile = get_original_tile(i, j, t)
    elif layer == 'comp':
        tile = get_compressed_tile(i, j, t)
    else:
        return arr

    for j in range(im_size):
        for i in range(im_size):
            oi = round((xs[j*im_size+i] - origin[0]) / 10.0)
            oj = round((origin[1] - ys[j*im_size+i]) / 10.0)

            if oi < 0 or oj < 0 or oi > 399 or oj > 399:
                continue
            
            #arr[j,i,:] = tile[oj,oi,:]  
            arr[j,i] = tile[oj,oi]  
            
    return arr

#app = Flask(__name__)

@app.route('/wms')
def wms():

    print(request.url)

    service = request.args.get('service')
    if service != 'WMS':
        return "Malformed request: only WMS service implemented", 400
    
    req = request.args.get('request')
    if req != 'GetMap':
        return "Malformed request: only GetMap requests implemented", 400

    layer = request.args.get('layers', default='comp')
    
    bbox = request.args.get('bbox').split(',')
    if len(bbox) != 4:
        return "Malformed request: bbox must have 4 values", 400
    bbox = [float(p) for p in bbox]
   
    t_idx = request.args.get('time', type=int, default=None)
    if t_idx is None:
        return "Malformed request: request need a time", 400

    width = int(request.args.get('width'))
    height = int(request.args.get('height'))
    srs = request.args.get('srs').lower()

    im = bbox2tile(bbox, layer, 1, t_idx, width, srs)
    im = np.uint8(cm.summer_r(im)*255)
    
    #im = np.clip(im, 0, 0.3643)
    #im *= 700
    #im = im.astype(np.uint8)

    out = io.BytesIO()
    imageio.imwrite(out, im, format='png') 
    out.seek(0)

    return send_file(out, mimetype='image/png')


@app.route('/')
def root():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8080', debug=False)
