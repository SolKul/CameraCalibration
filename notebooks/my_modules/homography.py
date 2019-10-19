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
import ransac


# +
def normalize(points):
    """列方向を一つの座標とみなし、最後の行が1になるように、最後の行で割り、正規化する"""
    return points/points[-1]

def make_homog(points):
    return np.vstack((points,np.ones((1,points.shape[1]))))

# +
# import numpy as np
# from matplotlib import pyplot as plt

# from sklearn import linear_model, datasets


# n_samples = 1000
# n_outliers = 50


# X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
#                                       n_informative=1, noise=10,
#                                       coef=True, random_state=0)

# # Add outlier data
# np.random.seed(0)
# X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
# y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# # Fit line using all data
# lr = linear_model.LinearRegression()
# lr.fit(X, y)

# # Robustly fit linear model with RANSAC algorithm
# ransac = linear_model.RANSACRegressor()
# ransac.fit(X, y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# # Predict data of estimated models
# line_X = np.arange(X.min(), X.max())[:, np.newaxis]
# line_y = lr.predict(line_X)
# line_y_ransac = ransac.predict(line_X)

# # Compare estimated coefficients
# print("Estimated coefficients (true, linear regression, RANSAC):")
# print(coef, lr.coef_, ransac.estimator_.coef_)

# lw = 2
# plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
#             label='Inliers')
# plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
#             label='Outliers')
# plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
# plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
#          label='RANSAC regressor')
# plt.legend(loc='lower right')
# plt.xlabel("Input")
# plt.ylabel("Response")
# plt.show()

# +

def H_from_points(fp, tp):
    if fp.shape != tp.shape:
        raise RuntimeError('number of points don\'t match')

    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1))+1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = C1 @ fp

    m = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1))+1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = C2 @ tp

    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0, i], -fp[1, i], -1, 0, 0, 0,
                  tp[0, i]*fp[0, i], tp[0, i]*fp[1, i], tp[0, i]]
        A[2*i+1] = [0, 0, 0, -fp[0, i], -fp[1, i], -1,
                    tp[1, i]*fp[0, i], tp[1, i]*fp[1, i], tp[1, i]]

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    H = np.linalg.inv(C2) @ (H@C1)

    return H/H[2, 2]


# -

class RansacModel:
    """ http://www.scipy.org/Cookbook/RANSAC のransac.pyを用いて
    ホモグラフィーを当てはめるためのクラス """
    def __init__(self,debug=False):
        self.debug = debug
        
    def fit(self, data):
        """ 4つの対応点にホモグラフィーを当てはめる """
        # H_from_points() を当てはめるために転置する
        data = data.T
        # 元の点
        fp = data[:3,:4]
        # 対応点
        tp = data[3:,:4]
        # ホモグラフィーを当てはめて返す
        return H_from_points(fp,tp)
    
    def get_error( self, data, H):
        """ すべての対応にホモグラフィーを当てはめ、各変換点との誤差を返す。"""
        data = data.T
        # 元の点
        fp = data[:3]
        # 対応点
        tp = data[3:]
        # fpを変換
        fp_transformed = H @ fp
        # 同次座標を正規化
        nz = np.nonzero(fp_transformed[2])
        for i in range(3):
            fp_transformed[i][nz] = fp_transformed[i][nz]/fp_transformed[2][nz]
            
        # 1点あたりの誤差を返す
        return np.sqrt( np.sum((tp-fp_transformed)**2,axis=0) )


def H_from_ransac(fp,tp,model,maxiter=1000,match_threshold=10):
    """ RANSACを用いて対応点からホモグラフィー行列Hをロバストに推定する
    (ransac.py は http://www.scipy.org/Cookbook/RANSAC を使用)
    入力: fp,tp (3*n 配列) 同次座標での点群 """
    # 対応点をグループ化する
    data = np.vstack((fp,tp))
    # Hを計算して返す
    H,ransac_data = ransac.ransac(data.T,model,4,maxiter,match_threshold,10,
                                  return_all=True)
    return H,ransac_data['inliers']
