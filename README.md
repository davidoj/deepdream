# simpledream

This repository contains a reorganisation of the deepdream ipython notebook to use with a class of students who are not familiar with programming.

The deepdream notebook complements the 
Google Research [blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html) about Neural Network art.
See [original gallery](https://photos.google.com/share/AF1QipPX0SCl7OzWilt9LnuQliattX4OUCj_8EP65_cTVnBmS1jnYgsGQAieQUc1VQWdgQ?key=aVBxWjhwSzg2RjJWLWRuVFBBZEN1d205bUdEMnhB) for more examples.

You can view "dream.ipynb" directly on github, or clone the repository, 
install dependencies listed in the notebook and play with code locally.

It'll be interesting to see what imagery people are able to generate
using the described technique. If you post images to Google+,
Facebook, or Twitter, be sure to tag them with #deepdream so other
researchers can check them out too.


##Dependencies
This notebook is designed to have as few dependencies as possible:
* Standard Python scientific stack: [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [PIL](http://www.pythonware.com/products/pil/), [IPython](http://ipython.org/). Those libraries can also be installed as a part of one of the scientific packages for Python, such as [Anaconda](http://continuum.io/downloads) or [Canopy](https://store.enthought.com/).
* [Caffe](http://caffe.berkeleyvision.org/) deep learning framework ([installation instructions](http://caffe.berkeleyvision.org/installation.html)).
* Google [protobuf](https://developers.google.com/protocol-buffers/) library that is used for Caffe model manipulation.
