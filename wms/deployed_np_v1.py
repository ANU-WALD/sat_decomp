import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided
from pyproj import Transformer
import io
import imageio
from datetime import datetime
from flask import send_file

from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.get_bucket("sentinel2_act")

layers = {"tc": [[0,1,2],4],"fc1": [[3,1,2],2],"fc2": [[3,1,0],2],"ndvi": [[0,3],1]}

aea_proj = "epsg:3577"
wm_proj = "epsg:3857"

def geotransform(i,j):
    return (1506645+i*4000,10,0,-3932965-j*4000,0,-10)

def tiles(x,y):
    return int((x-1506645)/4000.), int((-3932965-y)/4000.)

def bbox2tiles(bbox, im_size):
    pixel_size = ((bbox[2] - bbox[0]) / im_size, (bbox[3] - bbox[1]) / im_size)

    tile_xs = np.arange(bbox[0], bbox[2], pixel_size[0])
    tile_ys = np.arange(bbox[3], bbox[1], -pixel_size[1])
    tile_xs, tile_ys = np.meshgrid(tile_xs, tile_ys)
    transformer = Transformer.from_crs(wm_proj, aea_proj)
    aea_xs, aea_ys = transformer.transform(tile_xs.T, tile_ys.T)

    i0,j0=tiles(aea_xs.min(),aea_ys.max())
    i1,j1=tiles(aea_xs.max(),aea_ys.min())
    
    return (i0,i1,j0,j1,aea_xs,aea_ys)

def conv2d(a, b):
    a = as_strided(a,(len(a),a.shape[1]-len(b)+1,a.shape[2]-b.shape[1]+1,len(b),
                      b.shape[1],a.shape[3]),a.strides[:3]+a.strides[1:])
    return np.tensordot(a, b, axes=3)

def compose_tile(i,j,t,bands=[0,1,2]):
    
    blob = bucket.blob(f"{j:02d}_{i:02d}_coeffsnp.npy")
    f = io.BytesIO(blob.download_as_string())
    f.seek(0)
    coef = np.load(f).reshape(24,7,-1)[:,bands,t:t+1]
    
    blob = bucket.blob(f"{j:02d}_{i:02d}_meanf32.npy")
    f = io.BytesIO(blob.download_as_string())
    f.seek(0)
    mean = np.load(f).reshape(400,400)
    
    blob = bucket.blob(f"{j:02d}_{i:02d}_basenp.npy")
    f = io.BytesIO(blob.download_as_string())
    f.seek(0)
    base = np.load(f)
    
    blob = bucket.blob(f"{j:02d}_{i:02d}_kernp.npy")
    f = io.BytesIO(blob.download_as_string())
    f.seek(0)
    ker = np.load(f)
    
    res = np.einsum('ik,kj->ji', np.tanh(conv2d(base, ker)).reshape(-1,24), coef.reshape(24,-1)).reshape(-1,400,400)
    mean = mean + res
    return np.moveaxis(mean, 0, -1)

def get_tile(bbox,im_size,bands,t):
    i0,i1,j0,j1,aea_xs,aea_ys = bbox2tiles(bbox,im_size)
    arr = np.zeros((im_size,im_size,len(bands)), dtype=np.float32)
    for i in range(i0, i1+1):
        for j in range(j0, j1+1):
            im = compose_tile(j,i,t,bands)
            geot = geotransform(i,j)

            vi = np.round((aea_xs - geot[0]) / geot[1]).astype(np.int64)
            vi[(vi>399) | (vi<0)] = -1
            vj = np.round((aea_ys - geot[3]) / geot[5]).astype(np.int64)
            vj[(vj>399) | (vj<0)] = -1

            for pj in range(im_size):
                for pi in range(im_size):
                    if vi[pi,pj] < 0 or vj[pi,pj] < 0:
                        continue
                    arr[pj,pi] = im[vj[pi,pj],vi[pi,pj]]
                    
    return arr

def wms(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    if request.args and 'service' in request.args:
    	service = request.args.get('service')
    else:
        return "Malformed request: only WMS requests implemented", 400
    
    if service != 'WMS':
        return "Malformed request: only WMS requests implemented", 400

    if request.args and 'request' in request.args:
        req = request.args.get('request')
    else:
        return "Malformed request: only WMS requests implemented", 400
        
    if req != 'GetMap':
        return "Malformed request: only GetMap and GetCapabilities requests implemented", 400

    if request.args and 'bbox' in request.args:
        bbox = request.args.get('bbox').split(',')
    if len(bbox) != 4:
        return "Malformed request: bbox must have 4 values", 400
    bbox = [float(p) for p in bbox]

    if request.args and 'time' in request.args:
        try:
            date = datetime.strptime(request.args.get('time'), "%Y-%m-%dT%H:%M:%S.%fZ")
        except:
            return "Malformed date in request", 400
    else:
        date = datetime(2018,1,1)

    t = (date.year-2018)*365+date.timetuple().tm_yday

    if request.args and 'layers' in request.args:
        layer = request.args.get('layers')
    else:
        return "Malformed request: only WMS requests implemented", 400

    if layer not in layers:
        return f"Layer {layer} not recognised", 400
    
    width = int(request.args.get('width'))
    height = int(request.args.get('height'))
    srs = request.args.get('srs').lower()

    arr = get_tile(bbox,width,layers[layer][0],t)

    if layer != "ndvi":
        arr = (np.clip(arr, a_min=0, a_max=1.0/layers[layer][1])*layers[layer][1]*255).astype(np.uint8)
    else:
        arr = (arr[:,:,1]-arr[:,:,0])/(arr[:,:,1]+arr[:,:,0])
        arr = np.clip(arr, a_min=0, a_max=1)
        arr = (plt.cm.summer_r(arr)*255).astype(np.uint8)

    png_file = io.BytesIO()
    imageio.imwrite(png_file, arr, format='png')
    png_file.seek(0)

    return send_file(png_file, mimetype='image/png')
