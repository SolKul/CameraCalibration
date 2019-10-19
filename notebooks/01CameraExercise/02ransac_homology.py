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
#query画像とtrain画像について
#akazeでマッチングし、
#ransacによって外れ値を除去してHomology行列を算出する。
# -

import numpy as np
import cv2
import matplotlib.pyplot as plt

#自作モジュール
import image_processing as ip
# import desc_val as dv
# import camera
# import feature_detection as fd
# import homography

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

# +
# Initiate AKAZE detector
akaze = cv2.AKAZE_create()

# key pointとdescriptorを計算
kp1, des1 = akaze.detectAndCompute(img_query, None)
kp2, des2 = akaze.detectAndCompute(img_train, None)

#matcherとしてflannを使用。
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm         = FLANN_INDEX_LSH,
                   table_number      = 6,  
                   key_size          = 12,     
                   multi_probe_level = 1) 
search_params = dict(checks = 50)

# ANNで近傍２位までを出力
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2,k=2)

# store all the good matches as per Lowe's ratio test.
#2番めに近かったkey pointと差があるものをいいkey pointとする。
good_matches = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

#descriptorの距離が近かったもの順に並び替え
good_matches = sorted(good_matches, key = lambda x:x.distance)
        
#結果を描写
img_result = cv2.drawMatches(
    img_query, kp1, img_train, kp2, good_matches[:10], None, flags=2)
ip.show_img(img_result, figsize=(20, 30))

# +
#ransacによって外れ値を除去してHomology行列を算出する。
#opencvの座標は3次元のarrayで表さなければならないのに注意

#matchingの最小値を設定
MIN_MATCH_COUNT = 10

if len(good_matches)>MIN_MATCH_COUNT:
    #matching点の座標を取り出す
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    #ransacによって外れ値を除去
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    #query画像の高さ、幅を取得し、query画像を囲う長方形の座標を取得し、
    #それを算出された変換行列Mで変換する
    #変換した長方形をtrain画像に描写
    h,w = img_query.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img_train_drawn = cv2.polylines(img_train,[np.int32(dst)],True,(255,100,0),3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT))
    matchesMask = None
    
#結果を描写
num_draw=50

draw_params = dict(
#     matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask[:num_draw], # draw only inliers
                   flags = 2)

img_result_2 = cv2.drawMatches(img_query,kp1,img_train_drawn,kp2,good_matches[:num_draw],None,**draw_params)

ip.show_img(img_result_2,figsize=(20,30))
ip.imwrite('ransac_match.jpg',img_result_2)
