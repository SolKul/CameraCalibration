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

# +
#動画から、フレームを切り出し、
#立方体を描写し
#連番画像として保存する
# -

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#自作モジュール
import image_processing as ip
import camera
import homography


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


cap_file.get(cv2.CAP_PROP_FRAME_COUNT)

# +
img_query=ip.imread('CalibrationImage/query.JPG')
#画像を縮小
img_query=img_query[900:2400,2200:3700,:]
ip.show_img(img_query,show_axis=True)

K = camera.calculate_camera_matrix_w_sz(sz=(6000, 4000),lens='PZ')
K[0,2]=img_query.shape[1]/2
K[1,2]=img_query.shape[0]/2

cap_file = cv2.VideoCapture('./CalibrationImage/material.MP4')
print(type(cap_file))
print("suceed open video:",cap_file.isOpened())
width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_file.get(cv2.CAP_PROP_FPS)
total_frame=int(cap_file.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap_file.set(cv2.CAP_PROP_POS_FRAMES,0))

# VideoWriter を作成する。
# fourcc = cv2.VideoWriter_fourcc(*'X264')
# writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width,height))

# z=0の平面上の辺の長さ0.2の立方体の3Dの点
box = cube_points([0, 0, 0.1], 0.1)

im_edit=[]

for i in range(1700):
    file_name='output{:0=5}.jpg'.format(i)
    file_path=os.path.join('./output',file_name)
    
    ret, frame = cap_file.read()

    homography_matrix, mask=homography.compute_rasac_homology(img_query,frame,show_detail=False)
    
    if(np.all(homography_matrix==None)):
        ip.imwrite(file_path,frame)
        continue

    # 第1の画像の底面の正方形を射影する
    # カメラオブジェクトを定義
    #平行移動でz軸に-1を定義しているのは
    #立方体を手前に移動させるため
    #X→x_q→x_tとしているため、z軸での移動は
    #x_tでは拡大縮小に見える。
    P = np.hstack((K, K @ np.array([[0], [0], [-5]])))
    cam1 = camera.Camera(P)

    # 最初の点群は、底面の正方形
    pts = homography.make_homog(box[:, :5])
    box_cam1 = cam1.project(pts)
    
#     img_for_edit=img_query.copy()
#     for i in range(5):
#         circle_pt=tuple([int(pt) for pt in box_cam1[:2,i]])
#         cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
#     ip.show_img(img_for_edit,show_axis=True)

    # cam1とHから第2のカメラ行列を計算する
    P2=homography_matrix @ cam1.P
    cam2 = camera.Camera(P2)

    #　P=K (R|t)なので
    #P[:,:3]は K R
    #Pの前半3列にKの逆行列をかければRが算出できる。
    #そのRは2次元の回転しか加味していないので、
    #整形して3次元の回転も加味するようにする
    A = np.linalg.inv(K) @ cam2.P[:,:3]
    A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
    cam2.P[:,:3] = K @ A


    # 第2のカメラ行列を使って射影する
    box_cam2 = cam2.project(homography.make_homog(box))

    box2_int=box_cam2.astype(int).T
    img_for_edit=frame.copy()
    cv2.polylines(img_for_edit,[box2_int[:,:2]],True,(0,0,255),thickness=10,lineType=cv2.LINE_AA)
    
#     ip.show_img(img_for_edit,show_axis=True,figsize=(10,10))

    ip.imwrite(file_path,img_for_edit)
#     writer.write(img_for_edit)

# writer.release()
# -

print(cap_file.set(cv2.CAP_PROP_POS_FRAMES,1700))
ret, frame = cap_file.read()
ip.show_img(frame)

total_frame

os.path.join('./out',file_name)

np.all(None==None)

cam2.P



# +

img_query=ip.imread('CalibrationImage/query.JPG')
#画像を縮小
img_query=img_query[600:2600,1500:4500,:]
ip.show_img(img_query,show_axis=True)
# -

cap_file = cv2.VideoCapture('./output.avi')
print(type(cap_file))
print("suceed open video:",cap_file.isOpened())
ret, frame = cap_file.read()
ip.show_img(frame)

width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_file.get(cv2.CAP_PROP_FPS)
# cv2.VideoWriter(filename, fourcc, fps, frameSize[, isCol

# VideoWriter を作成する。
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
writer = cv2.VideoWriter('output.avi', fourcc, fps, (width,height))

for frame in im_edit:
    writer.write(frame)

writer.release()

# +
img_query=ip.imread('CalibrationImage/query.JPG')
#画像を縮小
img_query=img_query[900:2400,2200:3700,:]
ip.show_img(img_query,show_axis=True)

K = camera.calculate_camera_matrix_w_sz(sz=(6000, 4000),lens='PZ')
K[0,2]=img_query.shape[1]/2
K[1,2]=img_query.shape[0]/2

cap_file = cv2.VideoCapture('./CalibrationImage/material.MP4')
print(type(cap_file))
print("suceed open video:",cap_file.isOpened())
width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_file.get(cv2.CAP_PROP_FPS)
total_frame=int(cap_file.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap_file.set(cv2.CAP_PROP_POS_FRAMES,0))
# -

print(cap_file.set(cv2.CAP_PROP_POS_FRAMES,909))
ret, frame = cap_file.read()
ip.show_img(frame)

homography_matrix, mask=homography.compute_rasac_homology(img_query,frame,show_detail=True)

# +
# z=0の平面上の辺の長さ0.2の立方体の3Dの点
box = cube_points([0, 0, 0.1], 0.1)

P = np.hstack((K, K @ np.array([[0], [0], [-4]])))
cam1 = camera.Camera(P)

# 最初の点群は、底面の正方形
# pts = homography.make_homog(box[:, :5])
# box_cam1 = cam1.project(pts)

#     img_for_edit=img_query.copy()
#     for i in range(5):
#         circle_pt=tuple([int(pt) for pt in box_cam1[:2,i]])
#         cv2.circle(img_for_edit,circle_pt,50,(255,255,255),thickness=-1)
#     ip.show_img(img_for_edit,show_axis=True)

# cam1とHから第2のカメラ行列を計算する
P2=homography_matrix @ cam1.P
cam2 = camera.Camera(P2)

#　P=K (R|t)なので
#P[:,:3]は K R
#Pの前半3列にKの逆行列をかければRが算出できる。
#そのRは2次元の回転しか加味していないので、
#整形して3次元の回転も加味するようにする
A = np.linalg.inv(K) @ cam2.P[:,:3]
A = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = K @ A


# 第2のカメラ行列を使って射影する
box_cam2 = cam2.project(homography.make_homog(box))

box2_int=box_cam2.astype(int).T
img_for_edit=frame.copy()
cv2.polylines(img_for_edit,[box2_int[:,:2]],True,(0,0,255),thickness=10,lineType=cv2.LINE_AA)

ip.show_img(img_for_edit,show_axis=True,figsize=(10,10))
# -

cam2.P
