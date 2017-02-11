# wiscfnc

## Setup
1. Clone the wiscfnc repo using `git clone https://github.com/dconathan/wiscfnc.git`
2. Setup the proper environment using anaconda: `conda create env -f wiscfnc_env.yaml`
3. Enter the environment: `source activate wiscfnc`
4. Finalize environment setup installing tensorflow and keras: 
```
$ conda install -c conda-forge tensorflow
$ pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps
```

## Loading data
See docstring for load.py for instructions.
