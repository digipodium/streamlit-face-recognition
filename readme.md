# steps to install
1. One shot install (requires anaconda)
```
conda env create -f environment.yml
```
if this fails, then try the step below


1.
```
conda create --prefix .env python=3.9 pip

```
### activate the environment in cmd shell

2.
```
conda install -c conda-forge dlib
```
3.
```
pip install opencv-contrib-python face-recognition streamlit imutils shutil
```

