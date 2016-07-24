# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

import validators
import requests
import shutil
import re

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

layerdict = {'c1' : 'conv1/7x7_s2',
             'p1' : 'pool1/3x3_s2',
             'c2' : 'conv2/3x3_reduce',
             'p2' : 'pool2/3x3_s2',
             'i3a': 'inception_3a/1x1',
             'i3b': 'inception_3b/pool',
             'p3' : 'pool3/3x3_s2',
             'i4a': 'inception_4a/1x1',
             'i4b': 'inception_4b/5x5',
             'i4c': 'inception_4c/3x3_reduce',
             'i4d': 'inception_4d/output_inception_4d/output_0_split_2',
             'i4e': 'inception_4e/output',
             'p4' : 'pool4/3x3_s2_pool4/3x3_s2_0_split_1',
             'i5a': 'inception_5a/1x1',
             'i5b': 'inception_5b/pool_proj'}

def getImgName(url):
    return re.findall('[-\w\d\(\)\_]+\.(?:jpg|gif|png|jpeg)',url)[0]

def showimage(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def openimage(filename):
    if validators.url(filename):
        fn = getImgName(filename)
        if not fn:
            raise ValueError("That doesn't seem to be an image I can deal with")
        response = requests.get(filename, stream=True)
        with open(fn, 'wb') as outfile:
            shutil.copyfileobj(response.raw, outfile)
        del response
        filename = fn

    return np.float32(PIL.Image.open(filename))

model_path = '/home/david/Dropbox/code/caffe/models/bvlc_googlenet/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, step_size=1.5, end='inception_4d/output_inception_4d/output_0_split_2', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''



    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)   


def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              layer = 'c2', clip=True, **step_params):
    # prepare base images for all octaves
    end = layerdict[layer]
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showimage(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])
