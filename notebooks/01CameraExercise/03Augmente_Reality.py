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
import cv2
import matplotlib.pyplot as plt

#自作モジュール
import image_processing as ip
import camera
import homography

# +
ratio=0.3

img_query=ip.imread('CalibrationImage/FeatureDetection00001.JPG')
#画像を縮小
img_query=cv2.resize(img_query,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
ip.show_img(img_query,show_axis=True)

img_train=ip.imread('CalibrationImage/FeatureDetection00003.JPG')
#画像を縮小
img_train=cv2.resize(img_train,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
ip.show_img(img_train,show_axis=True)
# -

homology_matrix, mask=homography.compute_rasac_homology(img_query,img_train,show_detail=True)


def cube_points(c,wid):
    """
    立方体を描画するための頂点のリストを生成する。
    最初の5点は底面の正方形であり、辺が繰り返されます
    Args:
        c (tuple):正方形の中心座標
    """
    p = []
    # 底面
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]-wid,c[2]-wid]) # 描画を閉じるため第一点と同じ
    # 上面
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) # 描画を閉じるため第一点と同じ
    # 垂直の辺
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    return np.array(p).T


img_query.shape

# +
#カメラキャリブレーション
K = camera.calculate_camera_matrix_w_sz(img_query.shape[:2])

# z=0の平面上の辺の長さ0.2の立方体の3Dの点
box = cube_points([0,0,0.1],0.1)
# 第1の画像の底面の正方形を射影する
#カメラオブジェクトを定義
P=np.hstack((K,K @ np.array([[0],[0],[-1]])))
cam1 = camera.Camera(P)
# -

P

box[:,:5]

pts=homography.make_homog(box[:,:5])
box_cam1=cam1.project(pts)

box_cam1

box_cam1[:2,0]

# +
# カメラキャリブレーション
K = camera.calculate_camera_matrix_w_sz(img_query.shape[1::-1],lens='SEL')

# z=0の平面上の辺の長さ0.2の立方体の3Dの点
box = cube_points([0, 0, 0.1], 0.1)
# 第1の画像の底面の正方形を射影する
# カメラオブジェクトを定義
P = np.hstack((K, K @ np.array([[0], [0], [1]])))
cam1 = camera.Camera(P)

# 最初の点群は、底面の正方形
pts = homography.make_homog(box[:, :5])
box_cam1 = cam1.project(pts)
# -

box_cam1

img_for_edit=img_query.copy()
for i in range(5):
    circle_pt=tuple([int(pt) for pt in box_cam1[:2,i]])
    cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit,show_axis=True)

# Hを使って第2の画像に点を変換する
box_trans = homography.normalize(homology_matrix @ box_cam1)

img_for_edit=img_train.copy()
for i in range(5):
    circle_pt=tuple([int(pt) for pt in box_trans[:2,i]])
    cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit,show_axis=True)

# cam1とHから第2のカメラ行列を計算する
P2=homology_matrix @ cam1.P
cam2 = camera.Camera(P2)

cam2.P[:,:3]

#　P=K (R|t)なので
#P[:,:3]は K R
#Pの前半3列にKの逆行列をかければRが算出できる。
#そのRをちょっときれいに整形しているっぽい
A = np.linalg.inv(K) @ cam2.P[:,:3]
A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = K @ A

# 第2のカメラ行列を使って射影する
box_cam2 = cam2.project(homography.make_homog(box))

box_cam2[:2,:]

tuple([int(pt) for pt in box_cam2[:2,i]])

box2_int=box_cam2.astype(int).T
box2_int[:3,:2]

img_for_edit=img_train.copy()
cv2.polylines(img_for_edit,[box2_int[:,:2]],True,(0,0,255),thickness=10,lineType=cv2.LINE_AA)
ip.show_img(img_for_edit,show_axis=True,figsize=(10,10))

cap_file = cv2.VideoCapture('./CalibrationImage/material.MP4')
print(type(cap_file))
print("suceed open video:",cap_file.isOpened())
ret, frame = cap_file.read()
ip.show_img(frame)
print(cap_file.set(cv2.CAP_PROP_POS_FRAMES,0))

# +
ratio=0.3

img_query=ip.imread('CalibrationImage/query.JPG')
#画像を縮小
# im_for_show=cv2.resize(img_query,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
img_query=img_query[900:2400,2200:3700,:]
ip.show_img(img_query,show_axis=True)

#画像を縮小
im_for_show2=cv2.resize(frame,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
ip.show_img(im_for_show2,show_axis=True)
# -

homology_matrix, mask=ransac_homology.compute_rasac_homology(img_query,frame,show_detail=True)

# +
# カメラキャリブレーション
K = camera.calculate_camera_matrix_w_sz(img_query.shape[1::-1],lens=None,f_orig=(6000,4188))

# z=0の平面上の辺の長さ0.2の立方体の3Dの点
box = cube_points([0, 0, 0.1], 0.1)
# 第1の画像の底面の正方形を射影する
# カメラオブジェクトを定義
P = np.hstack((K, K @ np.array([[0], [0], [1]])))
cam1 = camera.Camera(P)

# 最初の点群は、底面の正方形
pts = homography.make_homog(box[:, :5])
box_cam1 = cam1.project(pts)

img_for_edit=img_query.copy()
for i in range(5):
    circle_pt=tuple([int(pt) for pt in box_cam1[:2,i]])
    cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit,show_axis=True)

# +
# cam1とHから第2のカメラ行列を計算する
P2=homology_matrix @ cam1.P
cam2 = camera.Camera(P2)

#　P=K (R|t)なので
#P[:,:3]は K R
#Pの前半3列にKの逆行列をかければRが算出できる。
#そのRをちょっときれいに整形しているっぽい
A = np.linalg.inv(K) @ cam2.P[:,:3]
A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = K @ A

# 第2のカメラ行列を使って射影する
box_cam2 = cam2.project(homography.make_homog(box))

box2_int=box_cam2.astype(int).T
img_for_edit=frame.copy()
cv2.polylines(img_for_edit,[box2_int[:,:2]],True,(0,0,255),thickness=10,lineType=cv2.LINE_AA)

ip.show_img(img_for_edit,show_axis=True,figsize=(10,10))

# +
ratio=0.3

img_query=ip.imread('CalibrationImage/query.JPG')
#画像を縮小
# im_for_show=cv2.resize(img_query,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
img_query=img_query[900:2400,2200:3700,:]
ip.show_img(img_query,show_axis=True)

cap_file = cv2.VideoCapture('./CalibrationImage/material.MP4')
print(type(cap_file))
print("suceed open video:",cap_file.isOpened())


im_edit=[]

print(cap_file.set(cv2.CAP_PROP_POS_FRAMES,0))

for i in range(10):
    ret, frame = cap_file.read()
    ip.show_img(frame)

    homology_matrix, mask=homography.compute_rasac_homology(img_query,frame)

    # カメラキャリブレーション
    K = camera.calculate_camera_matrix_w_sz(img_query.shape[1::-1],lens=None,f_orig=(6000,4188))

    # z=0の平面上の辺の長さ0.2の立方体の3Dの点
    box = cube_points([0, 0, 0.1], 0.1)
    # 第1の画像の底面の正方形を射影する
    # カメラオブジェクトを定義
    P = np.hstack((K, K @ np.array([[0], [0], [1]])))
    cam1 = camera.Camera(P)

    # 最初の点群は、底面の正方形
    pts = homography.make_homog(box[:, :5])
    box_cam1 = cam1.project(pts)

    # cam1とHから第2のカメラ行列を計算する
    P2=homology_matrix @ cam1.P
    cam2 = camera.Camera(P2)

    #　P=K (R|t)なので
    #P[:,:3]は K R
    #Pの前半3列にKの逆行列をかければRが算出できる。
    #そのRをちょっときれいに整形しているっぽい
    A = np.linalg.inv(K) @ cam2.P[:,:3]
    A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
    cam2.P[:,:3] = K @ A

    # 第2のカメラ行列を使って射影する
    box_cam2 = cam2.project(homography.make_homog(box))

    box2_int=box_cam2.astype(int).T
    img_for_edit=frame.copy()
    cv2.polylines(img_for_edit,[box2_int[:,:2]],True,(0,0,255),thickness=10,lineType=cv2.LINE_AA)

im_edit.append(img_for_edit)
# -

ip.show_img(img_query)

img_query.shape

cap_file.release()

# +
print(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))

print(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(cap_file.get(cv2.CAP_PROP_FPS))

print(cap_file.get(cv2.CAP_PROP_FRAME_COUNT))

print(cap_file.get(cv2.CAP_PROP_FRAME_COUNT) / cap_file.get(cv2.CAP_PROP_FPS))
# -

print(cap_file.get(cv2.CAP_PROP_POS_FRAMES))
print(cap_file.get(cv2.CAP_PROP_POS_MSEC))


ret, frame = cap_file.read()

# +
print(ret)

print(type(frame))

print(frame.shape)
# -

print(cap_file.get(cv2.CAP_PROP_POS_FRAMES))
print(cap_file.get(cv2.CAP_PROP_POS_MSEC))
print(1 / cap_file.get(cv2.CAP_PROP_FPS) * 1000)

ip.show_img(frame)

from IPython.display import HTML

print(1920/3,1080/3)

HTML(data='''<video alt="test" controls width="640">
                <source src="./CalibrationImage/C0009.MP4" type="video/mp4" />
             </video>''')

img_for_edit=img_train.copy()
for i in range(17):
    circle_pt=tuple([int(pt) for pt in box_cam2[:2,i]])
    cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit,show_axis=True)

# テスト：z=0上の点を射影を変換すると同じになるはず
point_on_query = np.array([1, 1, 0, 1]).T
point_on_train = homology_matrix @ cam1.P @ point_on_query
print(homography.normalize(point_on_train))
print(cam2.project(point_on_query))

# テスト：z=0上の点を射影を変換すると同じになるはず
point = np.array([1,1,0,1]).T
print(homography.normalize(point_on_query))
print(cam2.project(point))

P @ pt_3d

# +
#任意のワールド座標を定義
pt_3d=np.array([0,0,720,1])

pt_2d=cam1.project(pt_3d)

#プロットしてみる
img_for_edit=img_query.copy()
circle_pt=tuple([int(pt) for pt in pt_2d[:2]])
cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
ip.show_img(img_for_edit,show_axis=True)
# -

img_query.shape[-2:]

img_query.shape[1::-1]
