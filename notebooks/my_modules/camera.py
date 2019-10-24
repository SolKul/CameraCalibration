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
import numpy as np
from scipy import linalg

class Camera:
    '''
    カメラ行列Pを与えてカメラオブジェクトを作るモデル。
    
    Args:
        P (array):カメラ行列P=K[R|t]。
    '''
    def __init__(self,P):
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
        K,R=linalg.rq(self.P[:,:3])
        T=np.diag(np.sign(np.diag(K)))
        #K,Rの行列式を正にする
        self.K= np.dot(K,T)
        self.R= np.dot(T,R)
        self.t= np.linalg.inv(self.K) @ self.P[:,3]
        
        return self.K, self.R, self.t
    
    def center(self):
        """カメラ中心を計算して返す"""
        if self.c is not None:
            return self.c
        else:
            self.factor()
            self.c= - self.R.T @ self.t
            return self.c
    
def rotation_matrix(a):
    """ ベクトルaを軸に回転する3Dの回転行列を返す """
    R = np.eye(4)
    R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R

def calculate_camera_matrix_w_sz(sz,sz_orig=(6000,4000),lens='PZ',f_orig=(4188,4186)):
    """
    異なる解像度でのカメラの内部パラメータKを計算する関数
    Args:
        sz (int):扱う画像サイズ。もともとの画像から縮小していた場合など。
        sz_orig (int):(6000,4000)はα6000で24MPで撮影したときの解像度
        lens (str): 'PZ'か'SEL'
        f_orig (int):(5266,5150)はα6000でSEL18200で焦点距離18のときの焦点距離
        (4188,4186)はPZ1650でズーム16のときの焦点距離
    """
    if(lens=='PZ'):
        f_orig=(4188,4186)
    elif(lens=='SEL'):
        f_orig=(5266,5150)
    fx_orig,fy_orig=f_orig
    width,height=sz
    width_orig,height_orig=sz_orig
    fx=fx_orig * width /width_orig
    fy=fy_orig * height /height_orig
    K = np.diag([fx,fy,1])
    K[0,2]=0.5*width
    K[1,2]=0.5*height
    return K
# -

# カメラ行列の原理  
# まずピンホールカメラモデルについて理解  
# https://www.slideshare.net/ShoheiMori/ss-64994150  
#
# ついでに同次座標系について理解  
# http://zellij.hatenablog.com/entry/20120523/p1  
#
# 要は座標を扱うときはその1次元上で扱うと便利という話  
#
# そしてopencv公式
# https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html  
#
# ![カメラ行列の原理](https://docs.opencv.org/2.4/_images/pinhole_camera_model.png)

# ## カメラ行列について
#
# $\boldsymbol{x}$:画像上の2次元座標
# $$
# \\
# \boldsymbol{x}=\left(\begin{array}{c}
# x\\
# y\\
# 1
# \end{array}\right)\\
# \  \\
# $$
# $\boldsymbol{X}$:3次元座標  
# $$
# \\
# \boldsymbol{X}=\left(\begin{array}{c}
# X\\
# Y\\
# Z\\
# W
# \end{array}\right)\\
# \  \\
# $$
#
# ピンホールカメラを用いると、3Dの点Xから画像上の点 x（どちらも同次座標で表現）への射影は
# 次の式で表せます。
# $$\lambda \boldsymbol{x}=P\boldsymbol{X}=K(R|t)\boldsymbol{X}$$
# まずワールド座標系での3Dの点Xをカメラを中心とした座標系に
# $$
# (R|t)\boldsymbol{X}
# $$
# で射影します。  
# $(R|t)$は$3\times4$なのでこの射影で同次座標系ではなく、普通の座標になるっぽい  
#   
#   
# そして$K$で画像上の2次元座標に射影される  
#   
#
#
# $P=K(R|t)$  
# $K$:内部パラメータ  
# $$
# \\
# K=\left(\begin{array}{ccc}
# f_x & s & c_x\\
# 0 & f_y & c_y\\
# 0 & 0 & 1
# \end{array}\right)\\
# \  \\
# $$
# $fx,fy$:焦点距離。厳密にはレンズの焦点距離とは違うので注意。縦横で異なることがあるのでfx,fy別になっている。  
# $c_x,c_y$左端から、画像中心までの距離。$x$を左端からの座標系に戻すために必要  
# $R$:回転行列(3×3)  
# $t$:並行移動(3×1)  
# $P$:カメラ行列、3次元座標を画像上の2次元座標に射影する行列  
# $\lambda,W$:よくわからん

# 行列式と行列の積の関係
# $$
# C=AB\\
# |C|=|A||B|\\
# $$
# カメラ行列$K$とその対角成分の符号を取ったもの$T$
# $$
# \\
# K=\left(\begin{array}{ccc}
# f_x & s & c_x\\
# 0 & f_y & c_y\\
# 0 & 0 & 1
# \end{array}\right)\\
# \  \\
# \  \\
# T=\left(\begin{array}{ccc}
# sgn(f_x) & 0 & 0\\
# 0 & sgn(f_y) & 0\\
# 0 & 0 & sgn(1)
# \end{array}\right)\\
# \ \\
# \ \\
# sgn(|K|)=sgn(|T|)\\
# \ \\
# $$
# その積$K'$の行列式は必ず正
# $$
# K'=KT\\
# \ \\
# if\ |T|>0\\
# sgn(|K|)=sgn(|T|)>0\\
# |K'|=|K||T|>0\\
# \ \\
# else\ |T|<0\\
# sgn(|K|)=sgn(|T|)<0\\
# |K'|=|K||T|>0\\
# $$
