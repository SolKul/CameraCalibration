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

# %load_ext autoreload
# %autoreload

import numpy as np
import cv2
import matplotlib.pyplot as plt

#自作モジュール
import image_processing as ip
import desc_val as dv
import camera
import feature_detection as fd
import homography


def expand(image, ratio):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (2*w, 2*h), cv2.INTER_LANCZOS4) # 補間法も指定できる


# +
ratio=0.3

img_for_camera_calibration=ip.imread('CalibrationImage/FeatureDetection00001.JPG')
im1=expand(img_for_camera_calibration,ratio)
im1=im1[:int(4000*ratio),:int(6000*ratio)]
ip.show_img(im1,show_axis=True)

img_math_test=ip.imread('CalibrationImage/FeatureDetection00003.JPG')
im2=expand(img_math_test,ratio)
im2=im2[:int(4000*ratio),:int(6000*ratio)]
ip.show_img(im2,show_axis=True)
# + {}
akaze = cv2.AKAZE_create()

kp1, des1 = akaze.detectAndCompute(im1,None)
kp2, des2 = akaze.detectAndCompute(im2,None)

# Brute-Force Matcher生成
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None, flags=2)
ip.show_img(img3,figsize=(20,30))
# -

im_draw=im1.copy()
for i in range(10):
    m=matches[i]
    key_point_pt=kp1[m.queryIdx].pt
    key_point_pt=tuple([int(value) for value in key_point_pt])
    cv2.circle(im_draw,key_point_pt,100,(0,100,200),thickness=20)
ip.show_img(im_draw)
ip.imwrite('match_result_circle.jpg',im_draw)


# 対応点を同次座標の点に変換する関数
def convert_points(matches,kp1,kp2):
#     ndx = matches.nonzero()[0]
    fp=np.array([kp1[m.queryIdx].pt for m in matches])
    fp = homography.make_homog(fp.T)
#     ndx2 = [int(matches[i]) for i in ndx]
    tp=np.array([kp2[m.trainIdx].pt for m in matches])
    tp = homography.make_homog(tp.T)
    return fp,tp


# +
# ホモグラフィーを推定
model = homography.RansacModel()
fp,tp = convert_points(matches,kp1,kp2)
H,inlier=homography.H_from_ransac(fp,tp,model)

inlier
# -

inl_matches=[matches[i] for i in inlier]
img3 = cv2.drawMatches(im1, kp1, im2, kp2, inl_matches, None, flags=2)
ip.show_img(img3,figsize=(20,30))
ip.imwrite('inl_match.jpg',img3)

np.vstack((fp,tp))

# +
MIN_MATCH_COUNT=10

# Initiate AKAZE detector
akaze = cv2.AKAZE_create()

# find the keypoints and descriptors with AKAZE
kp1, des1 = akaze.detectAndCompute(im1,None)
kp2, des2 = akaze.detectAndCompute(im2,None)

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
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# +
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = im1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(im2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

# +
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask[:50], # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(im1,kp1,im2,kp2,good[:50],None,**draw_params)

ip.show_img(img3,figsize=(20,30))
ip.imwrite('inl_match.jpg',img3)
# -

len(good)

len(good)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

search_params

data=np.arange(24).reshape(6,-1)
# data[:3,:]
data

for i in range(10):
    m=matches[i]
    key_point_pt=kp1[m.queryIdx].pt
#     key_point_pt=tuple([int(value) for value in key_point_pt])
    

# +

tp=np.array([kp1[m.trainIdx].pt for m in matches[:100]])

# +
ratio=0.3

img_for_camera_calibration=mip.imread('CalibrationImage/ImageforCameraCalibration.jpg')
im1=expand(img_for_camera_calibration,ratio)
im1=im1[:int(4000*ratio),:int(6000*ratio)]
mip.show_img(im1,show_axis=True)
# -

img_math_test=mip.imread('CalibrationImage/MatchTest.jpg')
im2=expand(img_math_test,ratio)
im2=im2[:int(4000*ratio),:int(6000*ratio)]
mip.show_img(im2,show_axis=True)

# +
akaze = cv2.AKAZE_create()

kp1, des1 = akaze.detectAndCompute(im1,None)
kp2, des2 = akaze.detectAndCompute(im2,None)

# Brute-Force Matcher生成
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None, flags=2)
mip.show_img(img3,figsize=(20,30))
# -

from scipy import ndimage 

ndimage.affine_transform

mip.imwrite('match_result.jpg',img3)

mfd.compute_harris_ncc_and_plot(im1,im2)

mdv.desc_array(img_for_camera_calibration)

ratio=0.1
im_expand=expand(img_for_camera_calibration,ratio)
im_expand=im_expand[:int(4000*ratio),:int(6000*ratio)]
mip.show_img(im_expand,show_axis=True)

im1=cv2.cvtColor(im_expand,cv2.COLOR_BGR2GRAY)
mip.show_img(im1,show_axis=True)

brisk=cv2.BRISK_create()

kp1, des1 = brisk.detectAndCompute(im1,None)

len(kp1)

des1[0]



mip.show_img(img_for_camera_calibration)



akaze = cv2.AKAZE_create()

mip.imwrite('match_result.jpg',img3)

matches_s[2].distance

bf.getDefaultName()

im1 = cv2.cvtColor(img_for_camera_calibration,cv2.COLOR_BGR2GRAY)
mip.show_img(im1)

matches = bf.knnMatch

cv2.AKAZE_create()

cv2.__version__

sys.version

im=img_for_camera_calibration.copy()
im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
mip.show_img(im)

harrisim=mfd.compute_harris_response(im)

filtered_coords=mfd.search_harris_point(harrisim,min_dist=100)

mip.show_img(img_for_camera_calibration,figsize=(10,10))
plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')

len(filtered_coords)

desc=mfd.extract_pixels_near_coord(im,filtered_coords)
match_indices=mfd.extract_same_matches(desc,desc)
# mfd.extract_max_ncc_indices(desc,desc)

# +
im1=img_for_camera_calibration.copy()
im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
mip.show_img(im1)

im2=img_math_test.copy()
im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
mip.show_img(im2)
# -

wid =5
harrisim = mfd.compute_harris_response(im1,5)
filtered_coords1 = mfd.search_harris_point(harrisim,100)
d1 = mfd.extract_pixels_near_coord(im1,filtered_coords1,wid)

harrisim = mfd.compute_harris_response(im2,5)
filtered_coords2 = mfd.search_harris_point(harrisim,100)
d2 = mfd.extract_pixels_near_coord(im2,filtered_coords2,wid)

match_indices=mfd.extract_same_matches(d1,d2)

mfd.plot_matches(im1,im2,filtered_coords1,filtered_coords2,match_indices,figsize=(20,10))

im1=mip.imread('messi5.jpg')
mip.show_img(im1)
im2=mip.imread('JudeaPearl.jpg')
mip.show_img(im2)

im1.shape

np.zeros((height2-height1,*im1.shape[1:])).shape

im3 = mfd.concatenate_img_horiz(im,im)


filtered_coords

locs1=filtered_coords
locs2=filtered_coords

mfd.plot_matches(im,im,locs1,locs2,match_indices,figsize=(20,10))

im1=im
im3=mfd.concatenate_img_horiz(im,im)
mip.show_img(im3,figsize=(20,10))
width1 = im1.shape[1]
for i,m in enumerate(matchs_indices):
    if m>=0:
        plt.plot([locs1[i][1],locs2[m][1]+width1],[locs1[i][0],locs2[m][0]],'c')
plt.axis('off')

mip.show_img(im3,figsize=(20,10))

# +
# im1=im
# im2=im
height1 = im1.shape[0]
height2 = im2.shape[0]

dimension1=len(im1.shape)
dimension2=len(im2.shape)

if (dimension1==2) & (dimension2==2):
    if height1 < height2:
        im1_cvt = np.concatenate((im1,np.zeros((height2-height1,im1.shape[1]))),axis=0)
        im2_cvt = im2.copy()
    elif height1 > height2:
        im1_cvt = im1.copy()
        im2_cvt = np.concatenate((im2,np.zeros((height1-height2,im2.shape[1]))),axis=0)
else:
    if (dimension1==3) & (dimension2==2):
        im1_cvt = im1.copy()
        im2_cvt=cv2.cvtColor(im2,cv2.COLOR_GRAY2BGR)
    elif (dimension1==2) & (dimension2==3):
        im1_cvt=cv2.cvtColor(im1,cv2.COLOR_GRAY2BGR)
        im2_cvt = im2.copy()
    if height1 < height2:
        im1_cvt = np.concatenate((im1,np.zeros((height2-height1,*im1.shape[1:]))),axis=0)
        im2_cvt = im2.copy()
    elif height1 > height2:
        im1_cvt = im1.copy()
        im2_cvt = np.concatenate((im2,np.zeros((height1-height2,*im2.shape[1:]))),axis=0)

im_c=np.concatenate((im1_cvt,im2_cvt),axis=1).astype(np.uint8)
mip.show_img(im_c)
# -

height2-height1

np.zeros((height2-height1,*im1.shape[1:])).shape

im1_cvt.shape

im_c.min()

im2.shape

im1.shape

# +
desc1=desc
desc2=desc

n=len(desc1[0])

#対応点ごとの距離
d = -np.ones((len(desc1),len(desc2)))
for i in range(len(desc1)):
    for j in range(len(desc2)):
        d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
        d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
        ncc_value = np.sum(d1 * d2) / (n-1)
        if ncc_value > threshold:
            d[i,j] = ncc_value
ndx = np.argsort(-d)
matchs_indices = ndx[:,0]
#matchs_indicesは画像1と画像2の記述子のうち、
#画像1から画像2への相関関数が最大のもののindex
# -

d[:10,:10]

# +
desc=mfd.extract_pixels_near_coord(im,filtered_coords)
matches_12=mfd.extract_max_ncc_indices(desc,desc)
matches_21=mfd.extract_max_ncc_indices(desc,desc)
#matchs_12は画像1と画像2の記述子のうち、
#画像1から画像2への相関関数が最大のもののindex

ndx_12= np.where(matches_12 >= 0)[0]

# for n in ndx_12:
#     if matches_21[matches_12[n]] != n:
#         matches_12[n] = -1
# -

match21[matches_12[1]]

desc

desc=[]
wid=5
for coords in filtered_coords:
    patch=im[(coords[0]-wid):(coords[0]+wid),
                (coords[1]-wid):(coords[1]+wid)].flatten()
    desc.append(patch)


# +
threshold = 0.5
desc1=desc
desc2=desc.copy()
n=len(desc1[0])

d=-np.ones((len(desc1),len(desc2)))
for i in range(len(desc1)):
    for j in range(len(desc2)):
        d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
        d2 = (desc2[i] - np.mean(desc2[i])) / np.std(desc2[i])
        ncc_value = np.sum(d1 * d2) / (n-1)
        if ncc_value > threshold:
            d[i,j] = ncc_value
ndx = np.argsort(-d)
matchscores = ndx[:,0]
# -

 ndx

np.argsort([3,2,31],order=)

mfd.compute_harris_response(im)

threshold=0.1
corner_threshold=harrisim.max()*threshold
harrisim_t=(harrisim > corner_threshold) *1

coords = np.array(harrisim_t.nonzero()).T

[c for c in coords]

candidate_values = [harrisim[c[0],c[1]] for c in coords]

index = np.argsort(candidate_values)
min_dist=10
allowed_locations=np.zeros(harrisim.shape)
allowed_locations[min_dist:-min_dist,min_dist:-min_dist]=1

filtered_coords = []
for i in index:
    if allowed_locations[coords[i,0],coords[i,1]] == 1:
        filtered_coords.append(coords[i])
        allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                         (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
filtered_coords

mip.show_img(img_for_camera_calibration,figsize=(10,10))
plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')

import matplotlib.pyplot as plt

img=im
min_intens=img.min()
max_intens=img.max()
img_show=((img-min_intens)/(max_intens-min_intens)*255).astype(np.uint8)
plt.figure()
plt.imshow(img_show,cmap='gray')



im

plt.imshow(im)

mdv.desc_array(img_for_camera_calibration,globals())

mc.calculate_camera_matrix_w_sz(sz=(100,200))

img=ip.imread('01JudeaPearl.jpg')
ip.show_img(img)

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
