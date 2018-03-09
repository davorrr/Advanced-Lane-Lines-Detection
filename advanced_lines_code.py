# -*- coding: utf-8 -*-

"""
Created on Sat Jan 20 19:35:25 2018
@author: davor
"""

import glob
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
#import IPython.display as HTML



# Calibrating the camera
def calibrate(calibration_images):
    """This function is used to provide camera calibration parameters which
    are then used to undistort images. It should be called using glob library.
    Camera parameters are declared as global variables so that they can be 
    used in the entire script.
    """
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


def abs_sobel_thresh(img, orient='x', sobel_kernel=9, thresh=(20,150)):
    """ This function applies Sobel transform on the image and the thresholds it
    """

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

def hls_select(img, channel_select='s', thresh=(90,250)):
    """ Converts the image from BGR to HLS color space and then singles out one
    channel and thresholds it."""

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
    """ Undistorts images using camera calibration parameters"""
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist

def combined(sobel, saturated):
    """ Combines diffent thresholded binary images so that maximum amount of
    information can be pulled from them."""
    combined = np.zeros_like(sobel)
    combined[((sobel == 1) | (saturated == 1))] = 1
    
    return combined

def warp(image):
    """ Warps image from normal into birds-eye perspective so that the lane 
    lines can be modeled as a polinomial function."""
    global perspective_M
    src = np.float32([[592,445],[688,445],[0,719],[1279,719]])
    
    offset = 210
    dst = np.float32([[offset,0],[1280-offset,0],[offset,720],[1280-offset,720]])
    
    #dst = np.float32([[110,0],[980,0],[110,720],[980,720]])
    perspective_M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, perspective_M, (1280,720), flags=cv2.INTER_LINEAR)
    
    return warped

class Line():
    """ A class that keeps track of the characteristics of each line detection."""
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_fit = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

# IMAGE PIPELINE
    
# Reading in the calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

calibrate(images)


def pipeline(img):
    undist = undistort_image(img)
    sobel_x = abs_sobel_thresh(undist)
    saturation = hls_select(undist)
    combined_image = combined(sobel_x, saturation)
    binary_warped = warp(combined_image)
    
    return binary_warped


def first_frame_lines(img):
    """ This function detects the lane lines on the first image of the video.
    """
    ### FINDING LANE LINES
    binary_warped = pipeline(img)
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                      (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                      (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    left_line.current_fit = np.polyfit(lefty, leftx, 2)
    #left_fit = left_line.current_fit
    left_line.detected = True
    right_line.current_fit = np.polyfit(righty, rightx, 2)
    #right_fit = right_line.current_fit
    right_line.detected = True

    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img

def second_ord_poly(line, val):
    """ Simple function being used to help calculate distance from center.
    Only used within Detect Lines below. Finds the base of the line at the
    bottom of the image.
    """
    a = line[0]
    b = line[1]
    c = line[2]
    formula = (a*val**2)+(b*val)+c

    return formula


def detect_lines(img):
    """ Checks if the First Frame Lines function detected the lines on the 
    first frame. If not it calls it and if the lines are detected it uses them
    as initial information for the detection in the current frame. Afterwards 
    it fits the lines to the polinomial function and draws the detected lane 
    position on the frame. Also detects where the car is in the relation to the
    middle of the lane and what type of curvature it is driving at.
    """
    # Pull in the image
    binary_warped = pipeline(img)

    # Check if lines were last detected; if not, re-run first_lines
    if left_line.detected == False | right_line.detected == False:
        first_frame_lines(img)

    # Set the fit as the current fit for now
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # Again, find the lane indicators
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 20
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Set the x and y values of points on each line
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each again.
    # Similar to first_lines, need to try in case of errors
    # Left line first
    
    left_line.current_fit = np.polyfit(lefty, leftx, 2)
    left_fit = left_line.current_fit
    left_line.detected = True
    right_line.current_fit = np.polyfit(righty, rightx, 2)
    right_fit = right_line.current_fit
    right_line.detected = True
    

    # Generate x and y values for plotting
    fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    #left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx-margin, fity]))])
    #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx+margin, fity])))])
    #left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx-margin, fity]))])
    #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx+margin, fity])))])
    #right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Calculate the pixel curve radius
    y_eval = np.max(fity)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    avg_rad = round(np.mean([left_curverad, right_curverad]),0)
    rad_text = "Radius of Curvature = {}(m)".format(avg_rad)

    # Calculating middle of the image, aka where the car camera is
    middle_of_image = img.shape[1] / 2
    car_position = middle_of_image * xm_per_pix

    # Calculating middle of the lane
    left_line_base = second_ord_poly(left_fit_cr, img.shape[0] * ym_per_pix)
    right_line_base = second_ord_poly(right_fit_cr, img.shape[0] * ym_per_pix)
    lane_mid = (left_line_base+right_line_base)/2

    # Calculate distance from center and list differently based on left or right
    dist_from_center = lane_mid - car_position
    if dist_from_center >= 0:
        center_text = "{} meters left of center".format(round(dist_from_center,2))
    else:
        center_text = "{} meters right of center".format(round(-dist_from_center,2))
        
    # List car's position in relation to middle on the image and radius of curvature
    

    # Invert the transform matrix from birds_eye (to later make the image back to normal below)
    Minv = np.linalg.inv(perspective_M)

    # Plotting the lane
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(undistort_image(img), 1, newwarp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, center_text, (10,50), font, 1,(255,255,255),2)
    cv2.putText(result, rad_text, (10,100), font, 1,(255,255,255),2)
    
    return result
    
    
def process_image(img):
    """ Fuction used in processing the video."""
    return detect_lines(img)


# VIDEO PIPELINE

# Processing the video

left_line = Line()
right_line = Line()


output_video = 'output_video.mp4'
clip1 = VideoFileClip("project_video.mp4")

video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)