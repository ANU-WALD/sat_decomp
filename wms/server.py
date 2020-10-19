from flask import Flask
from flask import request
from flask import send_file
from flask import render_template
from flask import make_response

from pyproj import Proj, Transformer
import imageio
import numpy as np
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

app = Flask(__name__)


def get_tile(i, j, t):
    coeffs = np.load(pca_coeffs%(i,j))
    pcs = np.load(pca_pcs%(i,j)).reshape(-1,160000)
    mean = np.load(pca_mean%(i,j))
    stride = coeffs.shape[0]//7

    r = mean+(coeffs[None,t,:].dot(pcs)).reshape(400,400)
    g = mean+(coeffs[None,t+stride,:].dot(pcs)).reshape(400,400)
    b = mean+(coeffs[None,t+2*stride,:].dot(pcs)).reshape(400,400)
    
    return np.dstack((r,g,b))


def bbox2tile(bbox, band, t, im_size, proj):
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
            a = get_partial_tile(bbox, band, i, j, t, im_size, proj)
            if arr is None:
                arr = a
                continue
                
            arr += a
    
    return arr


def xy2ij(origin, pixel_size, x, y):
    i = round((x - origin[0]) / 10.0)
    j = round((origin[1] - y) / 10.0)
    
    return i, j
    
def get_partial_tile(bbox, b, i, j, t, im_size=256, proj=wgs84_proj):
    pixel_size = ((bbox[2] - bbox[0]) / im_size, (bbox[3] - bbox[1]) / im_size)

    arr = np.zeros((im_size,im_size,3), dtype=np.float32)
    
    lons = []
    lats = []
    for lat in np.arange(bbox[3], bbox[1], -pixel_size[1]):
        for lon in np.arange(bbox[0], bbox[2], pixel_size[0]):
            lons.append(lon)
            lats.append(lat)
    
    trans = Transformer.from_crs(proj, aaea_proj)
    xs, ys = trans.transform(lons, lats)

    print(xs[0], xs[-1], ys[0], ys[-1], len(xs))
   
    tile = get_tile(i, j, t)

    origin = (1506645.0+i*4000, -3932965.0-j*4000)

    for j in range(im_size):
        for i in range(im_size):
            oi, oj = xy2ij(origin, (10.0,-10.0), xs[j*im_size+i], ys[j*im_size+i])

            if oi > 399 or oj > 399:
                #arr[j,i] = np.nan
                #arr[j,i] = 0
                continue
            if oi < 0 or oj < 0:
                #arr[j,i] = np.nan
                #arr[j,i] = 0
                continue
            arr[j,i,:] = tile[oj,oi,:]  
            
    return arr

#app = Flask(__name__)

@app.route('/wms')
def wms():
    service = request.args.get('service')
    if service != 'WMS':
        return "Malformed request: only WMS requests implemented", 400

    req = request.args.get('request')
    if req == 'GetCapabilities':
        layers = [{'name': 'NDVI', 'title': 'Dynamic NDVI', 'abstract': 'AI generated'}]
        template = render_template('GetCapabilities.xml', layers=layers)
        response = make_response(template)
        response.headers['Content-Type'] = 'application/xml'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    if req != 'GetMap':
        return "Malformed request: only GetMap and GetCapabilities requests implemented", 400

    bbox = request.args.get('bbox').split(',')
    if len(bbox) != 4:
        return "Malformed request: bbox must have 4 values", 400
    bbox = [float(p) for p in bbox]

    width = int(request.args.get('width'))
    height = int(request.args.get('height'))
    srs = request.args.get('srs').lower()

    im = bbox2tile(bbox, 1, 0, width, srs)
    im = np.clip(im, 0, 0.3643)
    im *= 700
    im = im.astype(np.uint8)

    out = io.BytesIO()
    imageio.imwrite(out, im, format='png') 
    out.seek(0)

    return send_file(out, mimetype='image/png')


@app.route('/')
def root():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8080', debug=False)
