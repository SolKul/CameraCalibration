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

import numpy as np
import matplotlib.pyplot as plt
import cv2
#自作モジュール
import camera
import image_processing as ip
import desc_val as dv
import homography
import sfm

# +
# %matplotlib inline
show_detail=True
ratio=0.5
#画像を読み込み特徴点を計算する
im1 = ip.imread('./CalibrationImage/sfm_005.JPG')
#画像を縮小
im1=cv2.resize(im1,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)

im2 = ip.imread('./CalibrationImage/sfm_006.JPG')
#画像を縮小
im2=cv2.resize(im2,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)

ip.show_img(im1,show_axis=True)
ip.show_img(im2,show_axis=True)
plt.show()

# +
# K = np.array([[2394,0,932],[0,2398,628],[0,0,1]])
K = camera.calculate_camera_matrix_w_sz(im1.shape[1::-1],lens='PZ')

# Initiate AKAZE detector
akaze = cv2.AKAZE_create()

# key pointとdescriptorを計算
kp1, des1 = akaze.detectAndCompute(im1, None)
kp2, des2 = akaze.detectAndCompute(im2, None)

# matcherとしてflannを使用。
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)

# ANNで近傍２位までを出力
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
# 2番めに近かったkey pointと差があるものをいいkey pointとする。
good_matches = []
for i in range(len(matches)):
    if(len(matches[i])<2):
        continue
    m,n=matches[i]
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

        # descriptorの距離が近かったもの順に並び替え
good_matches = sorted(good_matches, key=lambda x: x.distance)

if(show_detail):
    # 結果を描写
    img_result = cv2.drawMatches(
        im1, kp1, im2, kp2, good_matches[:100], None, flags=2)
    ip.show_img(img_result, figsize=(20, 30))
    plt.show()
    print('queryのkp:{}個、trainのkp:{}個、good matchesは:{}個'.format(
        len(kp1), len(kp2), len(good_matches)))

num_inspect=1000
num_matches=len(good_matches)
#限界よりmatch点が多かったら
if num_matches<num_inspect:
    num_inspect=num_matches
x1=[]
x2=[]
for i in range(num_inspect):
    m=good_matches[i]
    k1=kp1[m.queryIdx]
    x1.append(k1.pt)
    
    k2=kp2[m.trainIdx]
    x2.append(k2.pt)

#同次座標系にし、K^-1を使って3次元座標にする。
x1=np.array(x1).T
x1=homography.make_homog(x1)

x2=np.array(x2).T
x2=homography.make_homog(x2)

x1n=np.linalg.inv(K) @ x1
x2n=np.linalg.inv(K) @ x2

#RANSACでEを推定
model = sfm.RansacModel()
E,inliers = sfm.F_from_ransac(x1n,x2n,model)

#カメラ行列を計算する
P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P2 = sfm.compute_P_from_essential(E)

#2つのカメラの前に点のある解を選ぶ
ind = 0
maxres = 0
for i in range(4):
    #triangulate inliers and compute depth for each camera
    #インライアを三角測量し、各カメラからの奥行きを計算する
    X=sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[i])
    d1 = (P1 @ X)[2]
    d2=(P2[i] @ X)[2]
    if np.sum(d1>0)+np.sum(d2>0) > maxres:
        maxres =np.sum(d1>0)+np.sum(d2>0)
        ind = i
        infront = (d1>0) & (d2 >0)
        
#インライアを三角測量し両方のカメラの正面に含まれていない点を削除します。
X=sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[ind])
X = X[:,infront]
# -

P2[ind]

x1.shape

# im1.shape
x2n[:,0] @ E @ x1n[:,0]

# +
# %matplotlib notebook
from mpl_toolkits.mplot3d import axes3d

fig= plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(X[0],X[1],X[2],'k.')
# plt.axis('off')

# +
# %matplotlib inline

#3Dの点群を射影変換する
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind])
x1p = cam1.project(X)
x2p = cam2.project(X)

#Kの正規化を戻し、2次元の画像上の座標とする
x1p = K @ x1p
x2p = K @ x2p

ip.show_img(im1)
plt.plot(x1p[0],x1p[1],'s')
plt.plot(x1[0],x1[1],'r*')

ip.show_img(im2)
plt.plot(x2p[0],x2p[1],'s')
plt.plot(x2[0],x2[1],'r*')
# -

np.argmax(X,axis=1)

X[:,7]

x1p[:,68]

X[:,228]=np.mean(X,axis=1)

# %matplotlib inline
ip.show_img(im1,figsize=(20,20))
plt.plot(x1p[0,68],x1p[1,68],'o')
plt.plot(x1[0,68],x1[1,68],'r.')

K
