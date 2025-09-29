import numpy as np
import cv2 as cv                
from .utils import show_img, show_imgs, scale_brightness

def split_grid(grid, n_rows=8, n_cols=16, debug=False):
    """Split grid of 7-segments into individual image dimensions
    
    This function is basically the opposite of gen_grid() in generate.py
    """
    
    # FYI, this assumes that each position is perfectly filled in our grid
    height = grid.shape[0] // n_rows
    width = grid.shape[1] // n_cols
    channels = grid.shape[2]
       
    tiles = np.zeros((n_rows*n_cols, height, width, channels), dtype=np.uint8)
    
    for i in range(n_rows):
        for j in range(n_cols):
            tiles[i*n_cols + j] = grid[i*height : i*height + height, j*width : j*width + width]
            
            if debug:
                show_img(f"tile {i*n_cols + j}", tiles[i*n_cols + j])
        
    return tiles

def calc_mosaic_size(image_shape, tile_shape):
    """figure out number of tiles we can fit in x and y dimension
    
    Example
    image(H, W) = (680, 615)
    tile(H, W) = (31, 17)
    we can fit 680 // 31 == 21 tiles in the y dimension
    we can fit 615 // 17 == 36 tiles in the x dimension
    so this is 21*36 == 756 tiles
    which means we have an output image of size (21*31, 36*17) == (651, 612)
    We have to round down, so that we have enough tiles to fit    
    """
    
    image_height_px = image_shape[0] # number of row pixels (image)
    image_width_px = image_shape[1] # number of column pixels (image)
    
    tile_height_px = tile_shape[0]
    tile_width_px = tile_shape[1]

    num_tiles_row = image_height_px // tile_height_px
    num_tiles_col = image_width_px // tile_width_px
    
    if num_tiles_row == 0 or num_tiles_col == 0:
        raise ValueError("Can't fit any tiles in mosaic. Try decreasing tile size, or increasing image size")

    return (num_tiles_row, num_tiles_col)

def center_crop_img(image, output_shape):
    """center crop image to number of tiles
    
    ------------------
    | | | | | | | | | |
    | | | | | | | | | |
    | | | | | | | | | |
    -----------------
    
    above is a (3 x 9) which represents original image
    we have tiles which are (2 x 5)
    if we start from top left
    
    -------------------
    |x|x|x|x|x| | | | |
    |x|x|x|x|x| | | | |
    | | | | | | | | | |
    -------------------
    
    but we should really center like this:
    
    -------------------
    | | |x|x|x|x|x| | |
    | | |x|x|x|x|x| | |
    | | | | | | | | | |
    -------------------
    
    
    Example
    image(H, W) = (680, 615)
    mosaic(H, W) = (651, 612)
    we should have equal number of cropped top/bottom, and left/right
    so 680-651 = 29 cropped rows
    615-612 = 3 cropped cols
    so crop image to -> 29 // 2 -> 14 offset
                     -> 3  // 2 -> 1 offset
    so add 14 to the row index, and subtract 14-1 from the end row index
    add 1 to the col index, and subtract 1-1 from the end col index
    
    cropped_row_range : [14, 665]
    cropped_col_range : [1, 613]
    need to add + 1 to end for python indexing reasons
    
    """
    image_height_px = image.shape[0]
    image_width_px = image.shape[1]
    output_height_px = output_shape[0]
    output_width_px = output_shape[1]
    
    row_offset = (image_height_px - output_height_px) // 2
    col_offset = (image_width_px - output_width_px) // 2
    
    cropped = image[row_offset : row_offset + output_height_px, col_offset: col_offset + output_width_px]
    
    return cropped


def tile(n_rows=8, n_cols=16,
         grid_img_path="segs.png",
         test_img_path="images/test/test_img_2_highres.jpg"):
    """Tile an image with candidate 7-segment display digit images

    Target image is first divided into blocks
    that are the same size as the tiles (extracted
    seven segment displays)

    1) Load and separate candidate tile images from grid
    2) Divide the target image into even number of tiles based on candidate tile sizes
    3) Calculate average brightness of all candidate tiles
    4) Pick the candidate tile with the closet brightness to a target image tile
    5) Move on to the next tile and repeat process until all tiles are placed
    """
        
    # candidate tiles as one big grid
    grid = cv.imread(grid_img_path, flags=cv.IMREAD_UNCHANGED)
    
    # invidiual candidate tiles accessed via 1st dimension
    tiles = split_grid(grid, n_rows, n_cols)
    
    # take in the image we are trying to convert
    image = cv.imread(test_img_path, flags=cv.IMREAD_UNCHANGED)
    
    # determine how many tiles we can fit in original image
    num_tiles_row, num_tiles_col = calc_mosaic_size(image.shape, tiles[0].shape)
    
    # create output image array
    n_channels = tiles.shape[-1]
    tile_height_px = tiles.shape[1]
    tile_width_px = tiles.shape[2]
    
    mosaic = np.zeros((num_tiles_row*tile_height_px, num_tiles_col*tile_width_px, n_channels), dtype=np.uint8)
    
    cropped = center_crop_img(image, mosaic.shape)
    
    show_imgs([(f"original image {image.shape}", image),
               (f"tiles[1] {tiles[1].shape}", tiles[1]),
               (f"mosaic grid {mosaic.shape}", mosaic),
               (f"cropped {cropped.shape}", cropped)])
    
    

    # Figure out brightness of each candidate tile
    # we iteratively average pixels from last dimension to first
    tiles_avg_brightness = np.average(tiles, axis=(3, 2, 1))
    
    # scale such that the smallest brightness corresponds to tile 0 (all segs off)
    # and the largest brightness corresponds to tile 127 (all segs on)
    tiles_avg_brightness = scale_brightness(tiles_avg_brightness)
    
    # Figure out brightness of each image block area
    image_avg_brightness = np.zeros((num_tiles_row, num_tiles_col), dtype=np.uint8)
    for i in range(num_tiles_row):
        for j in range(num_tiles_col):
            image_avg_brightness[i, j] = np.average(cropped[i*tile_height_px:tile_height_px*(i+1), j*tile_width_px:tile_width_px*(j+1)])
    show_img("Image average brightness", image_avg_brightness)
    
    # tiles_avg_brightness.shape == (128, )
    # image_avg_brightness.shape == (8, 12)
    # we reshape to do a single vector comparaision and use broadcasting 
    # to get which tile has the closet brightness
    diff = abs(image_avg_brightness.reshape(-1)[:, None] - tiles_avg_brightness)
    best_tiles = np.argmin(diff, axis=-1)
        
    best_tiles = best_tiles.reshape(num_tiles_row, num_tiles_col)
    # print(best_tiles)
    
    # (num_tiles_row, num_tiles_col, tile_height, tile_width, channels)
    extracted_tiles = tiles[best_tiles]
    
    # we transpose so that we can concatenate and stich together the 
    # tiles without writing explicit for loops
    extracted_tiles = extracted_tiles.transpose(0, 2, 1, 3, 4)
    extracted_tiles = extracted_tiles.reshape(num_tiles_row*tile_height_px, num_tiles_col*tile_width_px, n_channels)
    
    show_img("Mosaic output", extracted_tiles)
    
    return extracted_tiles        
        
if __name__ == "__main__":
    tile()


