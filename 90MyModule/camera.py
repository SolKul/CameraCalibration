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

# +
import numpy as np
from scipy import linalg

class camera:
    def __init__(self,P):
        '''カメラモデルP=K[R|t]を初期化する'''
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

def calculate_camera_matrix_w_sz(sz,sz_orig=(6000,4000),f_orig=(5266,5150)):
    '''異なる解像度でのカメラ行列を計算する関数
    (6000,4000)はα6000で24MPで撮影したときの解像度
    (5266,5150)はα6000でSEL18200で焦点距離18のときの焦点距離'''
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
# ![カメラ行列の原理](https://docs.opencv.org/2.4/_images/pinhole_camera_model.png)

# $$\lambda \boldsymbol{x}=P\boldsymbol{X}=K(R|t)\boldsymbol{X}$$
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
# $P=K(R|t)$  
# $K$:カメラ行列  
# $$
# \\
# K=\left(\begin{array}{ccc}
# f_x & s & c_x\\
# 0 & f_y & c_y\\
# 0 & 0 & 1
# \end{array}\right)\\
# \  \\
# $$
# $R$:回転行列(3×3)  
# $t$:並行移動(3×1)  
# $P$:Projection行列、3次元座標を画像上の2次元座標に射影する行列  
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
