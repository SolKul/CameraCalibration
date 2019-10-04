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
    
def show_img(img,figsize=(6,9),isBGR=True,show_as_it_is=False,show_axis=False):
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape)==3:
        if isBGR:
            img_cvt=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img_cvt=img
        plt.figure(figsize=figsize)
        plt.imshow(img_cvt)
    elif len(img.shape)==2:
        if show_as_it_is:
            img_show=img.astype(np.uint8)
            plt.figure(figsize=figsize)
            plt.imshow(img_show,cmap='gray',vmax=255)
        else:
            min_intens=img.min()
            max_intens=img.max()
            img_show=((img-min_intens)/(max_intens-min_intens)*255).astype(np.uint8)
            plt.figure(figsize=figsize)
            plt.imshow(img_show,cmap='gray')
    if not(show_axis):
        plt.axis('off')
