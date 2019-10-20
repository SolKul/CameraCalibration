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

im1, im2, points2D, points3D, corr, P=sfm.load_merton_data(show_detail=True)

#3Dの点を同次座標系にして射影する
X =homography.make_homog(points3D)
x=P[0].project(X)

points2D[0]

# +
#画像1の上に点を描写する
ip.show_img(im1,figsize=(6,4))
plt.plot(points2D[0][0],points2D[0][1],'*')
plt.show()

ip.show_img(im1,figsize=(6,4))
plt.plot(x[0],x[1],'*')
plt.show()
# -

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

import matplotlib
print(matplotlib.__version__)

# +
# %matplotlib notebook

fig = plt.figure(figsize=(6,4))
ax=fig.add_subplot(111,projection='3d')

# 3Dのサンプルデータを生成する。
X,Y,Z=axes3d.get_test_data(0.25)

# 3Dの点を描画する
ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1)

plt.show()
# -

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[0],points3D[1],points3D[2],c='k')

corr

# +
# %matplotlib inline
# 最初の２枚の画像の点のインデックス番号
ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)

# 座標値を取得し、同座標にする
x1 = points2D[0][:, corr[ndx, 0]]
x1 = homography.make_homog(x1)
x2 = points2D[1][:, corr[ndx, 1]]
x2 = homography.make_homog(x2)

#Fを計算する。
F=sfm.compute_fundamental(x1,x2)
#エピ極を計算する
e = sfm.compute_epipole(F)

#描写する
# plt.figure()
ip.show_img(im1)

#各行について適当に色を付けて描写する
for i in range(5):
    sfm.plot_epipolar_line(im1,F,x2[:,i],epipole=e,show_epipole=False)
plt.axis('off')

#描写する
# plt.figure()
ip.show_img(im2)

#各行について適当に色を付けて描写する
for i in range(5):
    plt.plot(x2[0,i],x2[1,i],'o')
plt.axis('off')

# +
# %matplotlib notebook

# 最初の２枚の画像の点のインデックス番号
ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)

# 座標値を取得し、同座標にする
x1 = points2D[0][:, corr[ndx, 0]]
x1 = homography.make_homog(x1)
x2 = points2D[1][:, corr[ndx, 1]]
x2 = homography.make_homog(x2)

Xtrue= points3D[:,ndx]
Xtrue= homography.make_homog(Xtrue)

#最初の3点を調べる
Xest=sfm.triangulate(x1,x2,P[0].P,P[1].P)
print(Xest[:,:3])
print(Xtrue[:,:3])

#描写する
fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(Xest[0],Xest[1],Xest[2],'ko')
ax.plot(Xtrue[0],Xtrue[1],Xtrue[2],'r.')
plt.show

# +
# %matplotlib inline
corr1=corr[:,0]#第一視点のインデックスを取り出す。
ndx3D=np.where(corr1>=0)[0]
ndx2D=corr1[ndx3D]

#見える点を選び、同次座標に変換
x = points2D[0][:, ndx2D]
x = homography.make_homog(x)
X = points3D[:, ndx3D]
X = homography.make_homog(X)

# Pを推定する
Pest= camera.Camera(sfm.compute_P(x,X))

#比較する
print(Pest.P / Pest.P[2,3])
print(P[0].P / P[0].P[2,3])

xest = Pest.project(X)

#描写する。
ip.show_img(im1)
plt.plot(x[0],x[1],'bo')
plt.plot(xest[0],xest[1],'r.')
plt.axis('off')
plt.show()
# -

print(im1.shape[0]/2,im1.shape[1]/2)

camera.calculate_camera_matrix_w_sz(im1.shape[1::-1],sz_orig=(932*2,628*2),lens=None,f_orig=(2394,2398))

# +
# %matplotlib inline
show_detail = True

K = np.array([[2394, 0, 932], [0, 2398, 628], [0, 0, 1]])

# 画像を読み込み特徴点を計算する
im1 = ip.imread('./Carl Olsson/Alcatraz_courtyard/San_Francisco_2313.jpg')
im2 = ip.imread('./Carl Olsson/Alcatraz_courtyard/San_Francisco_2314.jpg')

ip.show_img(im1)
ip.show_img(im2)
plt.show()

# K = camera.calculate_camera_matrix_w_sz(
#     im1.shape[1::-1], sz_orig=(932*2, 628*2), lens=None, f_orig=(2394, 2398))

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
    if(len(matches[i]) < 2):
        continue
    m, n = matches[i]
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

num_inspect = 1000
num_matches = len(good_matches)
# 限界よりmatch点が多かったら
if num_matches < num_inspect:
    num_inspect = num_matches
x1 = []
x2 = []
for i in range(num_inspect):
    m = good_matches[i]
    k1 = kp1[m.queryIdx]
    x1.append(k1.pt)

    k2 = kp2[m.trainIdx]
    x2.append(k2.pt)

# 同次座標系にし、K^-1を使って3次元座標にする。
x1 = np.array(x1).T
x1 = homography.make_homog(x1)

x2 = np.array(x2).T
x2 = homography.make_homog(x2)

x1n = np.linalg.inv(K) @ x1
x2n = np.linalg.inv(K) @ x2

# RANSACでEを推定
model = sfm.RansacModel()
E, inliers = sfm.F_from_ransac(x1n, x2n, model)

# カメラ行列を計算する
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_essential(E)

# 2つのカメラの前に点のある解を選ぶ
ind = 0
maxres = 0
for i in range(4):
    # triangulate inliers and compute depth for each camera
    # インライアを三角測量し、各カメラからの奥行きを計算する
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
    d1 = (P1 @ X)[2]
    d2 = (P2[i] @ X)[2]
    if np.sum(d1 > 0)+np.sum(d2 > 0) > maxres:
        maxres = np.sum(d1 > 0)+np.sum(d2 > 0)
        ind = i
        infront = (d1 > 0) & (d2 > 0)

# インライアを三角測量し両方のカメラの正面に含まれていない点を削除します。
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[ind])
X = X[:, infront]
# -

E=np.array([[  0.6230789 , -24.78075216,   6.19866097],
       [ 21.18761185,  -0.29148314, -25.76529314],
       [ -5.91487157,  22.13447932,   1.        ]])

# +
# %matplotlib notebook
from mpl_toolkits.mplot3d import axes3d

fig= plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(-X[0],X[1],X[2],'k.')
plt.axis('off')

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
plt.plot(x1p[0],x1p[1],'o')
plt.plot(x1[0],x1[1],'r.')
plt.savefig("Alcatraz_courtyard.png")

ip.show_img(im2)
plt.plot(x2p[0],x2p[1],'o')
plt.plot(x2[0],x2[1],'r.')
# -

data= np.vstack((x1n, x2n))
model.fit(data.T)

data[:3,:8].shape

from scipy import io
matdata = io.loadmat('./AlcatrazCourtyard/data.mat')

for k in matdata:
    print(k)

matdata

x1 = x1/x1[2]
mean_1 = np.mean(x1[:2], axis=1)
S1 = np.sqrt(2) / np.std(x1[:2])
T1 = np.array([[S1, 0, -S1*mean_1[0]],
               [0, S1, -S1*mean_1[1]],
               [0, 0, 1]])
xx1= T1 @ x1

np.std(xx1[:2],axis=1)**2

mean_1

np.std(xx1,axis=1)

x1[:2].shape

ndx3D

np.where(corr1>=0)

ndx.shape

compute_fundamental(x1,x1)

# +
x1 = np.arange(24).reshape(3, -1)
x2 = x1

n = x1.shape[1]
if x2.shape[1] != n:
    raise ValueError("Number of points don't match.")

# 方程式の行列を作成する
A = np.zeros((n, 9))
for i in range(n):
    A[i] = [x1[0, i]*x2[0, i], x1[0, i]*x2[1, i], x1[0, i]*x2[2, i],
            x1[1, i]*x2[0, i], x1[1, i]*x2[1, i], x1[1, i]*x2[2, i],
            x1[2, i]*x2[0, i], x1[2, i]*x2[1, i], x1[2, i]*x2[2, i]]

# 線形最小2乗法で計算する
U,S,V = np.linalg.svd(A)
F = V[-1].reshape(3,3)

#Fの制約
#最後の特異値を0にして階数2にする。
U,S,V = np.linalg.svd(F)
S[2] =0
F = U @ np.diag(S) @ V
# -

A=np.arange(9).reshape(3,-1)
A[2,2]=10
display(A)
U,S,V = np.linalg.svd(A)
dv.desc_array(U)
dv.desc_array(S)
dv.desc_array(V)

U

np.sqrt(np.sum(V[1,:]**2))

F

compute_fundamental(x1,x1)
