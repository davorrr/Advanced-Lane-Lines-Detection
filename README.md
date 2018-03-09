**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistortion_example.png "Undistorted image"
[image2]: ./output_images/undistortion_test.png "Undistorted test image"
[image3]: ./output_images/sobel_x.png "Sobel transform of the test image"
[image4]: ./output_images/saturation.png "Saturation component of the test image"
[image5]: ./output_images/combined.png "Combined Sobel and Saturation"
[image6]: ./output_images/source_destination.png "Perspective transform polygon"
[image7]: ./output_images/warped2.PNG "Warped thresholded image"
[image8]: ./output_images/final_frame.png "Lane detected"

[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function _calibrate()_ (lines 19 through 46 in the `advanced_lines_code.py`):
```python
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
```
In the code the calibration images were first loaded using glob library and the calibrated using the function above. In the function itself the "object points" were prepared first to hold (x, y, z) coordinates of the chessboard corners in the real world.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

After calibration the image was undistorted using the _undistort()_ function which is basically just a wrapper function for the OpenCV `cv2.undistort()` function:
```python
def undistort_image(image):
    """ Undistorts images using camera calibration parameters"""
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist
```
This is the obtained result:

![alt text][image1]

### Pipeline (single images)

In order to extract the lane data out of test images _pipline()_ function was written. 
```python
def pipeline(img):
    undist = undistort_image(img)
    sobel_x = abs_sobel_thresh(undist)
    saturation = hls_select(undist)
    combined_image = combined(sobel_x, saturation)
    binary_warped = warp(combined_image)
    
    return binary_warped
```
#### 1. Example of a distortion-corrected image

When we undistort the test image we get a result like this:
![alt text][image2]

#### 2. Thresholded binary image creation

First step was to create a thresholded binary image. To reach this several steps were made. First an undistorted image was passed through 2 functions: _abs_sobel_thresh()_ and _hls_select()_. First function is called in a way that it creates an thresholded binary image of a sobel tranfsorm along the x axis of the test image: 
```python
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
```
When applied it produces this image:
![alt text][image3]
Second function is called in a way that it selects the saturation out of the HLS color space:
```python
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
```
When we isolate the saturation component and create a binary image out of it we get this result:
![alt text][image4]

After these 2 steps we combine the both images unsing _combine()_ function into one in order to combine the different information extracted by them:
```python
def combined(sobel, saturated):
    """ Combines diffent thresholded binary images so that maximum amount of
    information can be pulled from them."""
    combined = np.zeros_like(sobel)
    combined[((sobel == 1) | (saturated == 1))] = 1
    
    return combined
```
Here is an example of the output in this step:
![alt text][image5]

#### 3. Perspective transform

After creating the thresholded binary image next step was to do a perspective transform from a normal into a birds-eye view. This was done using a function called _warp()_ which takes the thresholded binary image and outputs the birds-eye view:
```python
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
```
In the function source and destination points for the transform were hardcoded:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 592, 445      | 210, 0        | 
| 688, 445      | 1070, 0      |
| 0, 719     | 210, 720      |
| 1279, 719      | 1070, 720        |

These points drawn on a test image make a following polygon:
![alt text][image6]

After transforming the thresholded binary image we get this result:
![alt text][image7]




#### 4. Lane-line pixels identification and fitting 

After transforming the image it was necessary to extract lane position information as well as to mathematically model the lines by fitting them to the second order polynomial. This was done using the pocedured given in the tutorial.

The process was divided into two steps, coded into 2 different functions _first_frame_lines()_ and _detect_lines()_. First function detects the lines on the first frame of the video using a blind search. After the lines are detected we have a starting information that we then use in the rest of the video frames to detect lane lines using a detect_lines()_ function.

The process to detect lines in the first frame consits of creating a historgram of the bottom half of the image and than finding the positions of the peaks on the histogram left and right of the center, representing the lane lines. Afterward we do a search by dividing the image into 9 windows 80 pixels high and detecting non-zero pixel positions in these windows. If we detect more than 50 pixels we then center the window on that position and continue the search towards the top of the image. After finishing the search we concatenate the arrays of indices, extract line positions, fit the second order polynomial to the detected pixels, set the flags that the lines are detected on the first image and sve the information into the _Line()_ class created to store information between frames:
```python
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    left_line.current_fit = np.polyfit(lefty, leftx, 2)
    left_line.detected = True
    right_line.current_fit = np.polyfit(righty, rightx, 2)
    right_line.detected = True
```

In the _detect_lines()_ process is simplified because we already have initial information about line positions from the first frame. _detect_lines()_ itself first calls the _pipeline()_ function to process the frame and the depending on the status of lane line detection flags calls the _first_frame_lines()_ function or continues to detect lines by itself and to fit the polynomial. Except that the function also calculates the radius of curvature, position of the vechicle and plots the information on the frame along with the polygon representing the detected lane. 

Both function can be viewed in the [project's code](https://github.com/davorrr/CarND-Advanced-Lane-Lines-Detection/blob/master/advanced_lines_code.py)


#### 5. Curvature radius and position of the vechicle with respect to center 

Radius curvature and the position of the vechicle was calculated within the _detect_lines()_ function:
```python
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
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +                 left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
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

 
```

#### 6. Example image of the result plotted back down onto the road

This step was also implemented within the _detect_lines()_ function:
```python
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
```
This is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Final output video

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Pipline does it's job to detect the lanes in the project video but it is not very robust and it does not peform well in the challenge and harder challenge videos. Pipeline failes on nonhomogenous lanes with cracks in the pavement and more extreme curvatures. More time and effort is needed to enable it to detect the lanes in the challenge video as well as the harder challenge video. This could be done by adding more processing steps such as using the HSV color space, applying masks and also coding the solution to return out of error when pipeline fails to detect the line. Applying low pass filter's was also suggested as a solution. 

Other things to consider is that the approach used here would probably not perform well in driving situations involving steep hills because the perspective transform was calculated assuming a flat, even road. Other problems are any situations that would obstruct the camera view on the lanes, such as heavy traffic and adverse weather conditions such as heavy rain, snow (esspecialy), fog etc.

Possible solution to these problems to explore would be using deep neural network for lane lines detection instead of classical computer vision techniques. This approach would presumably reduce the need for preprocessing and also enable the car to generalise better to diverse road conditions, if trained correctly and with on an apropriate dataset, of course.
