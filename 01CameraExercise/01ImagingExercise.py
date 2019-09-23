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

# %load_ext autoreload
# %autoreload

# +
import sys
import numpy as np
import cv2
sys.path.append('../90MyModule/')

import image_processing as mip
import desc_val as mdv
import camera as mc
# -

mdv.desc_array(img_for_camera_calibration,globals())

mc.calculate_camera_matrix_w_sz(sz=(100,200))

img=ip.imread('01JudeaPearl.jpg')
ip.show_img(img)

img_for_camera_calibration=ip.imread('CalibrationImage/ImageforCameraCalibration.jpg')
ip.show_img(img_for_camera_calibration)

#本の左上にプロットしてみる
img_for_edit=img_for_camera_calibration.copy()
cv2.circle(img_for_edit,(2789,1140),50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit)

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

# $f_x=\frac{dx}{dX}dZ$  
# $f_y=\frac{dy}{dY}dZ$  

fx=width_in_image/153*720
fy=height_in_image/216*720
print('横方向の焦点距離f_x:%d\n縦方向の焦点距離f_y:%d' % (fx,fy))

K = np.array([[1000,0,500],[0,1000,300],[0,0,1]])
tmp = rotation_matrix([0,0,1])[:3,:3]
Rt = np.hstack((tmp,np.array([[50],[40],[30]])))
cam = Camera(np.dot(K,Rt))
print(K,'\n',Rt)
cam.factor()
print(cam.K,'\n',cam.R,'\n',np.linalg.det(cam.K),'\n',np.linalg.det(cam.R))
cam.center()

# +
P=K @ Rt
K2,R2=linalg.rq(P[:,:3])
T=np.diag(np.sign(np.diag(K2)))
if np.linalg.det(T)<0:
    T[1,1] *= -1

K2= K2@T
R= T@R2
print(K2)
print(R2)
# -

np.linalg.det(R2)

P=np.array([[1,2,3,4],[0,1,2,0],[0,1,3,4]])
c1=Camera(P)
X=np.array([2,3,-2,3])
c1.project(X)
c1.factor()
c1.K
np.linalg.det(c1.R)

from scipy.spatial.transform import Rotation as R
r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

r.as_dcm()
np.linalg.det(r.as_dcm())

K=np.array([[3,0,4],[0,4,3],[0,0,1]])
print(K2)
K2,R=linalg.rq(K @ r.as_dcm())
T=np.diag(np.sign(np.diag(K2)))
if np.linalg.det(T)<0:
    T[1,1] *= -1
print(np.linalg.det(K2 @ T))
print(np.linalg.det(T @ R))



K,R=linalg.rq(P[:,:3])
T=np.diag(np.sign(np.diag(K)))
np.linalg.det(T)
np.linalg.det(R)

T

c1.factor()

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
