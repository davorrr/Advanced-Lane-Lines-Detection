# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:35:25 2018

@author: davor
"""

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Calibrating the camera
def calibrate(calibration_images):
    # Arrays to store object points and image points from all the images
    global ret, mtx, dist, rvecs, tvecs
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image space
    
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

    for fname in calibration_images:
        img = mpimg.imread(fname)
        # Grayscaling
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Finding corners in the chessboard on the image
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
    # Finding camera calibration parameters       
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   (1280, 720), None, None)
    
    #return ret, mtx, dist, rvecs, tvecs


def abs_sobel_thresh(img, orient='x', sobel_kernel=9, thresh=(20,255)):
    # Calculating directional gradient and appying threshold

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Scaling is useful if we want our function to work on input images of
    # different scales for example on both png and jpg
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def hls_select(img, channel_select='s', thresh=(90,255)):

    # Converting to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Applying a threshold to channels
    if channel_select == 'h':
        H = hls[:,:,0]
        binary_output = np.zeros_like(H)
        binary_output[(H > thresh[0]) & (H <= thresh[1])] = 1
    elif channel_select == 'l': 
        L = hls[:,:,1]
        binary_output = np.zeros_like(L)
        binary_output[(L > thresh[0]) & (L <= thresh[1])] = 1
    elif channel_select == 's':
        S = hls[:,:,2]
        binary_output = np.zeros_like(S)
        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1  

    return binary_output


def undistort_image(image):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist


def combined(sobel, saturated):
    combined = np.zeros_like(sobel)
    combined[((sobel == 1) | (saturated == 1))] = 1
    
    return combined


def warp(image):
    src = np.float32([[592,445],[688,445],[0,719],[1279,719]])
    offset = 210
    dst = np.float32([[offset,0],[1280-offset,0],[offset,720],[1280-offset,720]])
    #dst = np.float32([[110,0],[980,0],[110,720],[980,720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (1280,720), flags=cv2.INTER_LINEAR)
    
    return warped

# IMAGE PIPELINE
    
# Reading in the calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Calibrating the camera
calibrate(images)

# Reading in the test image to undistort using the derived camera parameters
test_image = mpimg.imread('./test_images/test5.jpg')
  
#Undistorting test image
undist = undistort_image(test_image)

# Sobel x and Saturation
sobel_x = abs_sobel_thresh(undist)

# Finding saturation
saturation = hls_select(undist)

# Combining
combined_image = combined(sobel_x, saturation)

# Warping
binary_warped = warp(combined_image)


# Plotting the images
f, ax = plt.subplots(3, 2, figsize=(30, 15))
f.tight_layout()
ax[0][0].imshow(test_image)                                     #######
ax[0][0].set_title('Original Image', fontsize=50)
ax[0][1].imshow(undist) 
src = np.float32([[593,445],[687,445],[0,719],[1279,719]])
dst = np.float32([[110,0],[980,0],[110,720],[980,720]])                                        #######
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
ax[0][1].plot(x, y, color='r', linewidth=3)
ax[0][1].set_title('Undistorted image with perspective polygon', fontsize=50)
ax[1][0].imshow(sobel_x, cmap='gray')                           ######
ax[1][0].set_title('Thresholded Sobel', fontsize=50)
ax[1][1].imshow(saturation, cmap='gray')                          ######        
ax[1][1].set_title('Thresholded Saturation', fontsize=50)
ax[2][0].imshow(combined_image, cmap='gray')                            ######
#ax[2][0].plot(histogram, color='r')
ax[2][0].set_title('Combined binary images', fontsize=50)
ax[2][1].imshow(binary_warped, cmap='gray')                                         ######  
#ax[2][1].plot(left_fitx, ploty, color='yellow')
#ax[2][1].plot(right_fitx, ploty, color='yellow')
ax[2][1].set_title('Perspective transformed', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=1.8, bottom=0.)
