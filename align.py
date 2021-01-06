import sys
import random
import cv2
import numpy as np

from image import Image
from image import resize
from image import visualizeMatch

def match(img1, img2, default=True):
    des1 = img1.des
    des2 = img2.des
    if default:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        matched = []
        for m,n in matches:
            if m.distance < 0.5 * n.distance: # the smaller the ratio is , the correct the matches are
                matched.append(m)

        src = np.float32([img1.kp[m.queryIdx].pt for m in matched]).reshape(-1,2)
        dst = np.float32([img2.kp[m.trainIdx].pt for m in matched]).reshape(-1,2)
        return src, dst
    else:
        ratio = 0.8
        S = np.empty((0,2))
        D = np.empty((0,2))
        for i in range(des1.shape[0]):
            src = des1[i,:]
            candidates = []
            d1 = np.linalg.norm(src - des2[0,:])
            d2 = np.linalg.norm(src - des2[1,:])
            if d1 < d2:
                candidates.append([0, d1])
                candidates.append([1, d2])
            else:
                candidates.append([1, d2])
                candidates.append([0, d1])
            for j in range(2,des2.shape[0]):
                d = np.linalg.norm(src - des2[j,:])
                if d < candidates[0][1]:
                    candidates[0][1] = d
                    candidates[0][0] = j
                else:
                    if d < candidates[1][1]:
                        candidates[1][1] = d
                        candidates[1][0] = j
            
            if candidates[1][1]/candidates[0][1] > 0.8:
                S = np.concatenate((S, np.asarray(img1.kp[i].pt).reshape(1,2)))
                D = np.concatenate((D, np.asarray(img2.kp[candidates[0][0]].pt).reshape(1,2)))
        return S, D

# find similarity transform
def LeastSquare(src, dst):
    N = src.shape[0]

    # H*src = dst => src^T * H^T = dst^T
    # Convert to homogeneous coordinate by padding 1
    src = np.concatenate((np.transpose(src[:,:]), np.ones((1,N))), axis=0)
    dst = np.concatenate((np.transpose(dst[:,:]), np.ones((1,N))), axis=0)
    
    M, _, _, _ = np.linalg.lstsq(np.transpose(src), np.transpose(dst), rcond=None)
    M = np.transpose(M)
    return M

# https://blog.csdn.net/hudaliquan/article/details/52121832
def findHomography(src, dst, default=False):
    if src.shape[0] != 4:
        print("Error. Homography needs 4 pairs of points.")
    if default:
        return cv2.getPerspectiveTransform(src, dst)
    else:
        b = np.zeros((8,1))
        A = np.zeros((8,8))
        for i in range(4):
            b[  2*i, :] = dst[i,0]
            b[2*i+1, :] = dst[i,1]
            A[2*i:2*(i+1), :] = np.array( [[src[i,0], src[i,1], 1, 0, 0, 0, -dst[i,0]*src[i,0], -dst[i,0]*src[i,1]], \
                                           [0, 0, 0, src[i,0], src[i,1], 1, -dst[i,1]*src[i,0], -dst[i,1]*src[i,1] ]])
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        H = np.array([[x[0,0], x[1,0], x[2,0]], [x[3,0], x[4,0], x[5,0]], [x[6,0] , x[7,0], 1]])
        return H
        
def HomographyError(src, dst, M):
    x = (src[:,0]*M[0,0] + src[:,1]*M[0,1] +M[0,2])/(src[:,0]*M[2,0] + src[:,1]*M[2,1] +M[2,2])
    y = (src[:,0]*M[1,0] + src[:,1]*M[1,1] +M[1,2])/(src[:,0]*M[2,0] + src[:,1]*M[2,1] +M[2,2])
    error = np.power((x - dst[:,0]),2) + np.power((y - dst[:,1]),2)  
    return error

def RANSAC(src, dst, MAXITER=1000, default=False):
    
    N = src.shape[0]
    sampleNum = 4

    best_fit = float("Inf")
    best_inlier = 0
    threshold = 4
    best_M = np.ones((3,3))

    src = src[:,:]
    dst = dst[:,:]
    
    ### total error
    # for i in range(MAXITER):
    #     sample = random.sample(range(N),sampleNum)
    #     M = findHomography(src[sample], dst[sample], default=default)
    #     error = np.sum(HomographyError(src, dst, M))
    #     if error < best_fit:
    #         best_fit = error 
    #         best_M = M

    ### inlier
    for i in range(MAXITER):
        temp_inlier = 0
        sample = random.sample(range(N),sampleNum)
        M = findHomography(src[sample], dst[sample], default=default)
        error = HomographyError(src, dst, M)
        for j in range(src.shape[0]):
            if error[j] < threshold:
                temp_inlier += 1
        if temp_inlier > best_inlier:
            best_inlier = temp_inlier
            best_M = M

    return best_M

def transform(img1, img2, default=False):
    src, dst = match(img1, img2)
    # M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    # M = LeastSquare(src, dst)
    M = RANSAC(src, dst, default = default) # default: function provided by opencv
    return M