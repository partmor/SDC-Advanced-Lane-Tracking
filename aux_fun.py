import numpy as np
import cv2
import matplotlib.pyplot as plt

##########################
#  DISTORTION CORRECTION #
##########################

def undistort(img, camera_params):
    mtx = camera_params['mtx']
    dist = camera_params['dist']
    return cv2.undistort(img, mtx, dist, None, mtx)

#################
#  THRESHOLDING #
#################

def abs_sobel_thresh(channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # take the derivative in x or y given orient = 'x' or 'y'
    orient_param = (1, 0) if orient=='x' else (0, 1)
    sobel = cv2.Sobel(channel, cv2.CV_64F, orient_param[0], orient_param[1], ksize=sobel_kernel)
    # absolute value of the derivative or gradient
    abs_sobel = np.abs(sobel)
    # scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # binary mask where thresholds are met
    binary_res = np.zeros_like(channel)
    binary_res[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_res

def sobel_magnitude_thresh(channel, sobel_kernel=3, thresh=(0, 255)):
    # take the gradient in x and y separately
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1)
    # magnitude of the gradient
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # binary mask where thresholds are met
    binary_res = np.zeros_like(channel)
    binary_res[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_res

def dir_threshold(channel, sobel_kernel=3, thresh=(0, np.pi/2)):
    # take the gradient in x and y separately
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # binary mask where thresholds are met
    binary_res = np.zeros_like(channel)
    binary_res[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_res

def color_channel_thresh(channel, thresh=(0, 255)):
    binary_res = np.zeros_like(channel)
    binary_res[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary_res

def thresholding_pipeline(img_bgr):
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    h_ch, l_ch, s_ch = hls[:,:,0], hls[:,:,1], hls[:,:,2] 
    
    s_sobelx_bin = abs_sobel_thresh(s_ch, orient='x', sobel_kernel=3, thresh=(20, 100))
    s_bin = color_channel_thresh(s_ch, thresh=(180, 240))
    
    bin_res = s_bin | s_sobelx_bin
    return bin_res

##########################
#  PERSPECTIVE TRANSFORM #
##########################

def warp_image(img, src, dst):

    # compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

###################
#  LANE DETECTION #
###################

def find_lanes_sliding_window_hist(binary_warped_inp, get_viz=False):
    binary_warped = np.copy(binary_warped_inp)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped,)*3)*255 if get_viz else None
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
        if get_viz:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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
    
    if get_viz:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return leftx, lefty, rightx, righty, out_img

def find_lanes_near_previous(binary_warped_inp, l_fit_coeffs_prev, r_fit_coeffs_prev, margin=100 , get_viz=False):
    binary_warped = np.copy(binary_warped_inp)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (l_fit_coeffs_prev[0]*(nonzeroy**2) + l_fit_coeffs_prev[1]*nonzeroy + l_fit_coeffs_prev[2] - margin)) & (nonzerox < (l_fit_coeffs_prev[0]*(nonzeroy**2) + l_fit_coeffs_prev[1]*nonzeroy + l_fit_coeffs_prev[2] + margin))) 
    right_lane_inds = ((nonzerox > (r_fit_coeffs_prev[0]*(nonzeroy**2) + r_fit_coeffs_prev[1]*nonzeroy + r_fit_coeffs_prev[2] - margin)) & (nonzerox < (r_fit_coeffs_prev[0]*(nonzeroy**2) + r_fit_coeffs_prev[1]*nonzeroy + r_fit_coeffs_prev[2] + margin)))  

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Create an image to draw on and an image to show the selection window
    out_img = None
    if get_viz:
        out_img = np.dstack((binary_warped,)*3)*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return leftx, lefty, rightx, righty, out_img

def get_lane_fit_coeffs(lx_coords, ly_coords, rx_coords, ry_coords):
    # Fit a second order polynomial to each
    left_coeffs = np.polyfit(ly_coords, lx_coords, 2)
    right_coeffs = np.polyfit(ry_coords, rx_coords, 2)
    return left_coeffs, right_coeffs

############
#  METRICS #
############

def calculate_radius_in_meters(y_eval, fit_coeffs, x_meters_per_px=3.7/700, y_meters_per_px=30/720):
    alpha = (x_meters_per_px / y_meters_per_px)**2
    beta = x_meters_per_px / (y_meters_per_px**2)
    a, b = fit_coeffs[0], fit_coeffs[1]
    r = 1/beta*((1 + alpha*(2*a*y_eval + b)**2)**1.5) / np.absolute(2*a)
    return r

def calculate_offset_in_meters(img, left_fit_coeffs, right_fit_coeffs, x_meters_per_px=3.7/700):
    y_eval = img.shape[0]
    left_lane_bottom_x = left_fit_coeffs[0]*y_eval**2 + left_fit_coeffs[1]*y_eval + left_fit_coeffs[2]
    right_lane_bottom_x = right_fit_coeffs[0]*y_eval**2 + right_fit_coeffs[1]*y_eval + right_fit_coeffs[2]
    lane_midpoint_x = np.mean([left_lane_bottom_x, right_lane_bottom_x])
    img_bottom_midpoint_x = img.shape[1] / 2
    offset_px = img_bottom_midpoint_x - lane_midpoint_x
    offset_m = x_meters_per_px * offset_px
    return offset_m

###########
#  OUTPUT #
###########

def print_summary_on_original_image(undist_original_img, binary_warped,
                                    left_fitx_coords, right_fitx_coords, y_line_values,
                                    leftx_coords, lefty_coords, rightx_coords, righty_coords,
                                    left_radius, right_radius, offset,
                                    src, dst
                                   ):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero,)*3)
    
    l_lane_pixels_binary_warp = np.zeros_like(binary_warped).astype(np.uint8)
    l_lane_pixels_binary_warp[lefty_coords, leftx_coords] = 1
    r_lane_pixels_binary_warp = np.zeros_like(binary_warped).astype(np.uint8)
    r_lane_pixels_binary_warp[righty_coords, rightx_coords] = 1

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx_coords, y_line_values]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx_coords, y_line_values])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_original_img.shape[1], undist_original_img.shape[0])) 
    
    l_lane_pixels_binary = cv2.warpPerspective(l_lane_pixels_binary_warp, Minv, (undist_original_img.shape[1], undist_original_img.shape[0]))
    r_lane_pixels_binary = cv2.warpPerspective(r_lane_pixels_binary_warp, Minv, (undist_original_img.shape[1], undist_original_img.shape[0]))
    
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist_original_img, 1, newwarp, 0.3, 0)

    str_radius_info = 'Radius of Curvature = %0.1f m' % (np.mean([left_radius, right_radius]))
    str_offset_info = 'Vehicle is %0.2f m %s from the center' % (np.abs(offset), 'LEFT' if offset < 0 else 'RIGHT')
    cv2.putText(result, str_radius_info, (70,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, str_offset_info, (70,140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    result[l_lane_pixels_binary.nonzero()] = [255, 0, 0]
    result[r_lane_pixels_binary.nonzero()] = [0, 0, 255]
    return result

def pipeline(original_img_bgr, camera_params, src_vertices, dst_vertices):
    undist_img = undistort(original_img_bgr, camera_params)
    binary_img = thresholding_pipeline(undist_img)
    binary_warped = warp_image(binary_img, src_vertices, dst_vertices)
    leftx, lefty, rightx, righty, _ = find_lanes_sliding_window_hist(binary_warped, get_viz=False)
    left_fit, right_fit = get_lane_fit_coeffs(leftx, lefty, rightx, righty)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    r_left = calculate_radius_in_meters(y_eval, left_fit, 3.7/700, 30/720)
    r_right = calculate_radius_in_meters(y_eval, right_fit, 3.7/700, 30/720)
    offset = calculate_offset_in_meters(binary_warped, left_fit, right_fit, 3.7/700)
    res = print_summary_on_original_image(
        undist_img, binary_warped,
        left_fitx, right_fitx, ploty,
        leftx, lefty, rightx, righty,
        r_left, r_right, offset,
        src_vertices, dst_vertices
    )
    return res

############
#  HELPERS #
############

def draw_polygon_on_image_inplace(img, vertices, color=[0, 0, 255], thickness=5):
    for i in range(vertices.shape[0] - 1):
        cv2.line(
            img, 
            (vertices[i][0], vertices[i][1]), 
            (vertices[i + 1][0], vertices[i + 1][1]), 
            color=color, 
            thickness=thickness
        )   

def plot_3_channels(img, ch_names=('Ch. 1', 'Ch. 2', 'Ch. 3')):
    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,10))
    ax1.imshow(
        img[:,:,0],
        cmap='gray'
    )
    ax1.axis('off')
    ax1.set_title(ch_names[0])
    ax2.imshow(
        img[:,:,1],
        cmap='gray'
    )
    ax2.set_title(ch_names[1])
    ax2.axis('off')
    ax3.imshow(
        img[:,:,2],
        cmap='gray'
    )
    ax3.axis('off')
    ax3.set_title(ch_names[2])