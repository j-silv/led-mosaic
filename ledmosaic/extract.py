"""
 Description:
 
 Takes in a grid of possible combinations of a 7-segment display
 and generates individual image files for each possible combination

 - Some light thresholding is applied to the original image,
   such that no artifacts are present in the final processed image

 - Background color is added using the alpha channel as an index mask
 
 - Intensity of non-segment pixels is reduced to have a clearer segment image
 
 The grid image was found on Wikipedia: 
 https://upload.wikimedia.org/wikipedia/commons/d/d1/7-segment.svg
 
 Author: Justin Silver
 Date: 05/28/2023
"""

import numpy as np              # for array manipulation
import cv2 as cv                # for image stuff
import matplotlib.pyplot as plt # for plotting
import os


def load_img(img_path, resize=False, resize_factor=0.5):
    """Load the 7-segment grid image using OpenCV"""
    
    # flag needed to maintain alpha data
    seven_seg_grid_img = cv.imread(img_path, flags=cv.IMREAD_UNCHANGED)
    
    if resize == True:
        seven_seg_grid_img = cv.resize(seven_seg_grid_img,
                                       None,
                                       fx=resize_factor,
                                       fy=resize_factor,
                                       interpolation=cv.INTER_AREA)

    # OpenCV has BGRA ordering by default - fix RGB and BGR channels reverse order
    seven_seg_grid_img = cv.cvtColor(seven_seg_grid_img, cv.COLOR_BGRA2RGBA)
    
    return seven_seg_grid_img


def apply_img_thresholding(img):
    """Apply thresholding to raw image
    
    this is needed because there is some red in the upper edge of some digits
    if this is not done, an ugly red pixel artifact is seen after the
    intensity lowering code
    """
    
    (b, g, r, a) = cv.split(img)

    ret, r_thresh = cv.threshold(b, 250, 255, cv.THRESH_TRUNC)
    ret, g_thresh = cv.threshold(g, 250, 255, cv.THRESH_TRUNC)
    ret, b_thresh = cv.threshold(r, 250, 255, cv.THRESH_TRUNC)
    ret, a_thresh = cv.threshold(a, 250, 255, cv.THRESH_TRUNC)

    img = cv.merge((r_thresh, g_thresh, b_thresh, a_thresh))
    
    return img


def find_digits_in_grid(img):
    """Use a state machine to find location of digit pixels in grid"""

    # alpha channel is used to determine the start and end index of each digit in grid
    alpha = img[:, :, 3]

    # indexing into the grid to find the first digit (used for subsequent digits)
    digit_location = {
        'left': 0,
        'right': 0,
        'up' : 0,
        'down' : 0
    }

    # state machine to find each edge of the digit
    digit_search_state = 'NOTHING_FOUND'

    for row_idx in range(alpha.shape[0]):
        for col_idx in range(alpha.shape[1]):
            pixel = alpha[row_idx,col_idx]

            match digit_search_state:

                # because we are going by row from top to bottom, we will reach the top of the digit first
                case 'NOTHING_FOUND': 

                    if pixel > 0:
                        digit_search_state = 'DIG_TOP_EDGE_FOUND'
                        digit_location['up'] = row_idx
                        digit_location['left'] = col_idx # we just need to know the left top edge of the digit to compare later
                        break # we don't need to look at this row anymore 
                            # (it will just repeat the digits on the right)

                # now we need to get the left edge
                case 'DIG_TOP_EDGE_FOUND': 
                    
                    if pixel > 0:
                        if col_idx != digit_location['left']: # because we are at the slopeing part of the segment
                            digit_location['left'] = col_idx
                            break # we don't need to look at this row anymore, 
                                # cause we still haven't reached the left most edge of the digit
                        else:
                            # we had the same col location twice, so we are at an edge !
                            digit_search_state = 'DIG_LEFT_EDGE_FOUND'
                            digit_location['left'] = col_idx
                            # we don't need to break, since we can just go all 
                            # the way to the right to find the right edge

                case 'DIG_LEFT_EDGE_FOUND':
                    # not too robust, because we assume there is a white line across the alpha channel (ignoring black space in middle)
                    if pixel == 0:
                        digit_search_state = 'DIG_RIGHT_EDGE_FOUND'
                        digit_location['right'] = col_idx-1 # -1 because we found the black space, now we need to go one before to get the digit
                        break

                case 'DIG_RIGHT_EDGE_FOUND':
                    if pixel > 0:
                        digit_location['down'] = row_idx
                        break
                    elif col_idx == digit_location['right']:
                        digit_location['down'] = row_idx-1
                        digit_search_state = 'DIG_BOTTOM_EDGE_FOUND'
                        break

                case 'DIG_BOTTOM_EDGE_FOUND':
                    # we don't need to do anything here
                    break

                case _:
                    raise Exception("Invalid digit_search_state: ", digit_search_state)
        
        if (digit_search_state == 'DIG_BOTTOM_EDGE_FOUND'):
            break  # to break out of upper loop, we found the whole digit now

    return digit_location


def calculate_pixel_locations(digit_location, verbose=False):
    """Get some offset and digit size information from digit locations"""

    # number of pixels each digit's height takes up (+1 because top-left corner is at (0, 0) coordinates)
    digit_height = (digit_location['down'] - digit_location['up']) + 1 

    # number of pixels each digit's width takes up (+1 because top-left corner is at (0, 0) coordinates)
    digit_width = (digit_location['right']- digit_location['left']) + 1 

    # number of cols (x-axis pixels) to skip to get to the left-edge of the first digit
    col_offset = digit_location['left']

    # number of rows (y-axis pixels) to skip to get to the top-edge of the first digit
    row_offset = digit_location['up'] 

    if (verbose):
        print(f"digit_location: {digit_location}")
        print(f"digit_height: {digit_height}")
        print(f"digit_width: {digit_width}")
        print(f"col_offset: {col_offset}")
        print(f"row_offset: {row_offset}")
    
    return digit_height, digit_width, col_offset, row_offset


def add_special_offset(digit_row):
    """Add extra offset depending on specific digit row
    
    unfortunately the rows are not spaced out equally, whenever the .svg file was created
    this fixes it except for 3 special situations. There is one pixel off (on top) for digit_row:
    7, 6, 1
    It seems like the digit height is actually 1 pixel smaller than it should be... but that's ok. it is such
    a small error, it won't be too noticeable I bet
    Note that all the digits on this row will have the small pixel offset
    """

    additional_offset = 0
    if (digit_row == 0):
        additional_offset = 0
    elif (digit_row == 1):
        additional_offset = 0
    elif (digit_row == 2):
        additional_offset = 1
    elif (digit_row == 3):
        additional_offset = 1
    elif (digit_row == 4):
        additional_offset = 2
    elif (digit_row == 5):
        additional_offset = 2
    elif (digit_row == 6):
        additional_offset = 2
    elif (digit_row == 7):
        additional_offset = 2
    else:
        raise(Exception("Invalid digit row: ", digit_row))  
    
    return additional_offset


def apply_segment_thresholding(cropped_digit, digit_row, digit_col):
    """Threshold segment pixels and reduce brightness of non-segment pixels"""
    # R and B were again swapped
    cropped_digit_fixed_rgba = cv.cvtColor(cropped_digit[digit_row,digit_col,...], cv.COLOR_BGRA2RGBA)

    # R and B are again swapped
    (b, g, r, a) = cv.split(cropped_digit_fixed_rgba)

    segment_mask_r_gt_g = r > g # advanced boolean indices
    segment_mask_r_gt_b = r > b # advanced boolean indices

    segment_mask_r = np.logical_and(segment_mask_r_gt_g, segment_mask_r_gt_b)
    r[segment_mask_r] = 255 # this should already be the case
    g[segment_mask_r] = 0
    b[segment_mask_r] = 0

    # now we can reduce the intensity a little for the pixels that aren't included in this selection
    # basically this will be all the non-segment pixels
    r[np.logical_not(segment_mask_r)] = r[np.logical_not(segment_mask_r)]*0.25
    g[np.logical_not(segment_mask_r)] = g[np.logical_not(segment_mask_r)]*0.25
    b[np.logical_not(segment_mask_r)] = b[np.logical_not(segment_mask_r)]*0.25

    # remerge in the same order that we split
    cropped_digit_fixed_rgba = cv.merge((b, g, r, a))    
    
    return cropped_digit_fixed_rgba


def write_img(cropped_digit_fixed_rgba, digit_row, digit_col):
    """Add background and write image"""

    # https://stackoverflow.com/questions/53732747/set-white-background-for-a-png-instead-of-transparency-with-opencv
    background_mask = cropped_digit_fixed_rgba[...,3] == 0 # get indexes of where alpha channel is transparent
    cropped_digit_fixed_rgba[background_mask] = [0, 0, 0, 0] # use advanced boolean indexing and set background color
    cropped_digit_new_background = cv.cvtColor(cropped_digit_fixed_rgba, cv.COLOR_BGRA2BGR) # we don't need alpha channel anymore

    os.makedirs("images/extracted", exist_ok=True)
    cv.imwrite(f"images/extracted/{digit_row*16 + digit_col}.png", cropped_digit_new_background)


def extract_digit(img, digit_location, num_dig_rows=8, num_dig_cols=16, verbose=False):
    """Iterate through the grid image and write each image to an output file"""
    
    digit_height, digit_width, col_offset, row_offset = calculate_pixel_locations(digit_location, verbose)
    
    # 128 digits, with pixels of digit_height by digit_width, for all 4 channels
    cropped_digit = np.zeros((num_dig_rows, num_dig_cols, digit_height, digit_width, 4), dtype=np.uint8)

    for digit_row in range(cropped_digit.shape[0]):
        for digit_col in range(cropped_digit.shape[1]):

            additional_offset = add_special_offset(digit_row)
                                                                                             
            row_grid_start = row_offset*(2*digit_row + 1) + digit_height*(digit_row) + additional_offset
            row_grid_end = row_grid_start+digit_height

            col_grid_start = col_offset*(2*digit_col + 1) + digit_width*(digit_col)
            col_grid_end = col_grid_start + digit_width

            cropped_digit[digit_row,digit_col,...] = img[row_grid_start:row_grid_end,col_grid_start:col_grid_end,:]

            cropped_digit_fixed_rgba = apply_segment_thresholding(cropped_digit, digit_row, digit_col)

            write_img(cropped_digit_fixed_rgba, digit_row, digit_col)
            

    cropped_digit_grid = np.zeros((num_dig_rows*digit_height,num_dig_cols*digit_width,4),dtype=np.uint8)

    # there's probably a way to do this with numpy functions, I'm just not sure how
    # I need to concatenate in 2 directions at the same time
    for row in range(num_dig_rows) :
        cropped_digit_grid[row*digit_height:digit_height*(1+row),...] = np.concatenate(cropped_digit[row,...],axis=1)
        
    return cropped_digit_grid


def plot_img(img, cropped_digit_grid):
    """Use matplotlib to plot extraction results"""
    
    plt.figure(figsize=(20,8)) # just to get full-screen

    plt.subplot(221) # 2 row, 2 columns, 1st index
    plt.imshow(img)
    plt.title("7 segment RGBA")

    plt.subplot(222)
    plt.imshow(cropped_digit_grid, cmap='gray')
    plt.title("cropped_digit_grid")

    plt.subplot(223)
    cropped_digit_processed = cv.imread("images/extracted/1.png", flags=cv.IMREAD_UNCHANGED)
    cropped_digit_processed = cv.cvtColor(cropped_digit_processed, cv.COLOR_RGBA2BGRA)
    plt.imshow(cropped_digit_processed)
    plt.title("cropped_digit_processed[1]")

    plt.subplot(224)
    cropped_digit_processed = cv.imread("images/extracted/75.png", flags=cv.IMREAD_UNCHANGED)
    cropped_digit_processed = cv.cvtColor(cropped_digit_processed, cv.COLOR_RGBA2BGRA)
    plt.imshow(cropped_digit_processed)
    plt.title("cropped_digit_processed[75]")

    plt.show()

def extract():

    DEBUG = False
    NUM_DIG_ROWS = 8    # number of 7 segment digits in grid img per row
    NUM_DIG_COLS = 16   # number of 7 segment digits in grid img per column
    OUTPUT_IMAGE_ASPECT_RATIO = 70/123 # this is the default aspect ratio from the extracted image
    OUTPUT_IMAGE_HEIGHT = 30 # this is how big the output tile is in the Y axis
    OUTPUT_IMAGE_WIDTH = OUTPUT_IMAGE_HEIGHT*OUTPUT_IMAGE_ASPECT_RATIO
    INPUT_IMAGE_RESIZE_FACTOR = 0.5
    FLAG_RESIZE_ON = True
    assert(NUM_DIG_COLS*NUM_DIG_ROWS == 2**7) # 7 segments in on/off combination


    seven_seg_grid_img = load_img("images/seven_segment_grid.png", FLAG_RESIZE_ON, INPUT_IMAGE_RESIZE_FACTOR)
    seven_seg_grid_img = apply_img_thresholding(seven_seg_grid_img)
    digit_location = find_digits_in_grid(seven_seg_grid_img)    
    cropped_digit_grid = extract_digit(seven_seg_grid_img, digit_location, NUM_DIG_ROWS, NUM_DIG_COLS, DEBUG)
    plot_img(seven_seg_grid_img, cropped_digit_grid)
    




if __name__ == "__main__":
    extract()
