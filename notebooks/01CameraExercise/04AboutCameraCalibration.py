# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# カメラの校正について  
# 結果をcameraモジュールのcalculate_camera_matrix_w_sz関数にまとめる。

import numpy as np
import cv2
import matplotlib.pyplot as plt

#自作モジュール
import image_processing as ip
import desc_val as dv
import camera
import feature_detection as fd
import homography

# +
ratio=0.3

img_for_camera_calibration=ip.imread('CalibrationImage/ImageforCameraCalibration.jpg')
#画像を縮小
im1=cv2.resize(img_for_camera_calibration,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
ip.show_img(im1,show_axis=True)
# -

#本の左上にプロットしてみる
img_for_edit=img_for_camera_calibration.copy()
cv2.circle(img_for_edit,(2789,1140),50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit,show_axis=True)

# 写真上での四隅の座標  
# (左上の隅からのx座標、y座標)  
# 左上(2789,1140)   
# 右上(3910,1135)  
# 左下(2782,2685)  
# 右下(3902,2693)

#左上-左下→縦の長さ
height_in_image=np.sqrt((2789-2782)**2+(1140-2685)**2)
#左下-右下→横の長さ
width_in_image=np.sqrt((2783-3902)**2+(2685-2693)**2)
print('画像上で高さ%d、幅%d' % (height_in_image , width_in_image))

# 実際の本は  
# 縦216mm    
# 横153mm  
# 距離は720mm
#
# $f_x=\frac{dx}{dX}dZ$  
# $f_y=\frac{dy}{dY}dZ$  

fx=width_in_image/153*720
fy=height_in_image/216*720
print('横方向の焦点距離f_x:%d\n縦方向の焦点距離f_y:%d' % (fx,fy))


#検討の結果できた関数
def calculate_camera_matrix_w_sz(sz,sz_orig=(6000,4000),f_orig=(5266,5150)):
    """
    異なる解像度でのカメラ行列を計算する関数
    Args:
        sz (int):扱う画像サイズ。もともとの画像から縮小していた場合など。
        sz_orig (int):(6000,4000)はα6000で24MPで撮影したときの解像度
        f_orig (int):(5266,5150)はα6000でSEL18200で焦点距離18のときの焦点距離
    """
    fx_orig,fy_orig=f_orig
    width,height=sz
    width_orig,height_orig=sz_orig
    fx=fx_orig * width /width_orig
    fy=fy_orig * height /height_orig
    K = np.diag([fx,fy,1])
    K[0,2]=0.5*width
    K[1,2]=0.5*height
    return K


K=calculate_camera_matrix_w_sz((6000,4000))
display(K)

np.zeros((3,1))

P=np.hstack((K,np.zeros((3,1))))
cam_test=camera.Camera(P)
P

# +
#任意のワールド座標を定義
pt_3d=np.array([100,100,720,1])

pt_2d=cam_test.project(pt_3d)

#プロットしてみる
img_for_edit=img_for_camera_calibration.copy()
circle_pt=tuple([int(pt) for pt in pt_2d[:2]])
cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit,show_axis=True)

# +
ratio=0.3

img_for_camera_calibration=ip.imread('CalibrationImage/img_for_camera_calibration_PZ1650.JPG')
#画像を縮小
im1=cv2.resize(img_for_camera_calibration,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
ip.show_img(im1,show_axis=True)
# -

# PZ1650  
# 左上(2737,1004)  
# 右上(3538,1018)  
# 右下(3506,2148)  
#
# 実際の本は  
# 縦216mm    
# 横153mm  
# 距離は800mm

#右上-右下→縦の長さ
height_in_image=np.sqrt((3538-3506)**2+(1018-2148)**2)
#右上-左上→横の長さ
width_in_image=np.sqrt((3538-2737)**2+(1018-1004)**2)
print('画像上で高さ%d、幅%d' % (height_in_image , width_in_image))

fx=width_in_image/153*800
fy=height_in_image/216*800
print('横方向の焦点距離f_x:%d\n縦方向の焦点距離f_y:%d' % (fx,fy))
