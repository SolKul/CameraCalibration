# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
    
def ArToPlt(ar_img,figsize=(6,9)):
    plt.figure(figsize=figsize)
    n_img=cv2.cvtColor(ar_img,cv2.COLOR_BGR2RGB)
    plt.imshow(n_img)


# -

class Camera:
    def __init__(self,P):
        '''カメラモデルP=K[R|t]を初期化する'''
        self.P=P
        self.K=None 
        self.R=None
        self.t=None
        self.c=None

    def project(self,X):
        x=self.P @ X
        x=x/x[2]
        return x
    def factor(self):
        K,R=np.linalg.qr(self.P[:,:3])


P=np.array([[1,2,3,4],[0,1,2,0],[0,1,3,4]])
c1=Camera(P)
X=np.array([2,3,-2,3])
c1.project(X)

img=imread('DSC01805.JPG')

from PIL import Image

points=np.loadtxt('House/house.p3d').T
points=np.vstack((points,np.ones(points.shape[1])))
points.shape

P=np.hstack((np.eye(3),np.array([[0],[0],[0]])))
print(P.shape)
cam=Camera(P)
x=cam.project(points)
print(x.shape)

plt.figure()
plt.scatter(x[0,:],x[1,:],c='k')

with open('House/house.p3d') as f:
    print(f.read())
# points = loadtxt('House/house.p3d')

# + {"language": "bash"}
# tar -zxvf 3D.tar.gz
# -

ArToPlt(img)

ArToPlt(img,figsize=(10,15))

plt.imshow(img)

array = np.asarray(((img + 1) * 128).astype("i").transpose(1, 2, 0))
img = Image.fromarray(np.uint8(array))

import numpy as np
import cv2
