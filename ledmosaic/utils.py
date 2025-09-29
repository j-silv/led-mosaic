import cv2 as cv

def show_img(title, image):
    """Simple wrapper function around cv.imshow"""
    
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def show_imgs(images):
    """Same as show_img but creates multiple windows"""
    
    for (title, image) in images:
        cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def scale_brightness(arr, minval=0, maxval=255):
    """Apply linear scale to brightness"""
    
    scaled_arr = ( ((maxval - minval)*(arr - arr.min())) / (arr.max() - arr.min()) ) + minval
    
    return scaled_arr 

def calc_num_digits_display():
    """total number of digits for a certain example display
    
    (W:H) this uses the HDTV format. Another option is the 2.35:1 cinemascope aspect ratio. 
    Really the aspsect ratio we choose depends on the camera's aspect ratio

    Example display:

    ┌─────────────────────────────┐
    │ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ │
    │ │─│ │─│ │─│ │─│ │─│ │─│ │─│ │
    │ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ │
    │ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ │
    │ │─│ │─│ │─│ │─│ │─│ │─│ │─│ │ (DISPLAY_HEIGHT)
    │ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ │ 
    │ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ │
    │ │─│ │─│ │─│ │─│ │─│ │─│ │─│ │
    │ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ │
    └─────────────────────────────┘
        (DISPLAY_WIDTH)
    """

    DISPLAY_ASPECT_RATIO = 16/9
    DISPLAY_HEIGHT_INCH = 12*6 # the height of the art installation
    DISPLAY_WIDTH_INCH = DISPLAY_ASPECT_RATIO*DISPLAY_HEIGHT_INCH
    DISPLAY_AREA_SQ_INCH = DISPLAY_HEIGHT_INCH*DISPLAY_WIDTH_INCH

    PIXEL_HEIGHT_INCH = 0.75
    PIXEL_WIDTH_INCH = 0.5
    PIXEL_AREA_SQ_INCH = PIXEL_HEIGHT_INCH*PIXEL_WIDTH_INCH  # essentially the size of a single digit, includes the package size

    num_pixels_x = DISPLAY_AREA_SQ_INCH/PIXEL_AREA_SQ_INCH

    print(f"Number of pixels (digits): ${num_pixels_x}")
    
    return num_pixels_x