import numpy as np
import cv2 as cv                
from .utils import show_img, show_imgs, scale_brightness
import argparse

def split_grid(grid, rows=8, cols=16, debug=False):
    """Split grid of 7-segments into individual image dimensions
    
    This function is basically the opposite of gen_grid() in generate.py
    """
    
    # FYI, this assumes that each position is perfectly filled in our grid
    height = grid.shape[0] // rows
    width = grid.shape[1] // cols
    channels = grid.shape[2]
    
    # unsplit from rows*height, cols*width
    tiles = grid.reshape(rows, height, cols, width, channels)
    
    # reorder so that rows/cols combine after subsequent reshape
    tiles = tiles.transpose(0, 2, 1, 3, 4)
    
    tiles = tiles.reshape(rows*cols, height, width, channels)
    
    if debug:
        show_img(f"tiles[0]", tiles[0])
        show_img(f"tiles[-1]", tiles[-1])
        
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


def tile(*,
         rows=8,
         cols=16,
         grid_path="segs.png",
         image_path="images/test.jpg",
         debug=False):
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
    grid = cv.imread(grid_path, flags=cv.IMREAD_UNCHANGED)
    
    # invidiual candidate tiles accessed via 1st dimension
    tiles = split_grid(grid, rows, cols, debug)
    
    # take in the image we are trying to convert
    image = cv.imread(image_path, flags=cv.IMREAD_UNCHANGED)
    
    # determine how many tiles we can fit in original image
    num_tiles_row, num_tiles_col = calc_mosaic_size(image.shape, tiles[0].shape)
    
    # create output image array
    n_channels = tiles.shape[-1]
    tile_height_px = tiles.shape[1]
    tile_width_px = tiles.shape[2]
    
    cropped = center_crop_img(image, output_shape=(num_tiles_row*tile_height_px, num_tiles_col*tile_width_px, n_channels))
    
    if debug:
        show_imgs([(f"original image {image.shape}", image),
                   (f"tiles[1] {tiles[1].shape}", tiles[1]),
                   (f"cropped {cropped.shape}", cropped)])
    
    
    # Figure out brightness of each candidate tile
    # we iteratively average pixels from last dimension to first
    tiles_bright = np.average(tiles, axis=(3, 2, 1))
    
    # scale such that the smallest brightness corresponds to tile 0 (all segs off)
    # and the largest brightness corresponds to tile 127 (all segs on)
    tiles_bright = scale_brightness(tiles_bright)
    
    # same trick that we use in gen_grid/split_grid
    image_bright = cropped.reshape(num_tiles_row, tile_height_px, num_tiles_col, tile_width_px, n_channels)
    image_bright = image_bright.transpose(0, 2, 1, 3, 4)
    image_bright = np.average(image_bright, axis=(4, 3, 2))

    if debug:
        # image_bright is in floats, so we need to scale to range [0-1.0] for displaying
        show_img(f"Image average brightness per tile {image_bright.shape}", image_bright / image_bright.max())
    
    # tiles_bright.shape == (128, )
    # image_bright.shape == (8, 12)
    # we reshape to do a single vector comparaision and use broadcasting 
    # to get which tile has the closet brightness
    diff = abs(image_bright.reshape(-1)[:, None] - tiles_bright)
    best_tiles = np.argmin(diff, axis=-1)
        
    best_tiles = best_tiles.reshape(num_tiles_row, num_tiles_col)
    
    # (num_tiles_row, num_tiles_col, tile_height, tile_width, channels)
    mosaic = tiles[best_tiles]
    
    # we transpose so that we can concatenate and stich together the 
    # tiles without writing explicit for loops
    mosaic = mosaic.transpose(0, 2, 1, 3, 4)
    mosaic = mosaic.reshape(num_tiles_row*tile_height_px, num_tiles_col*tile_width_px, n_channels)
    
    if debug:
        show_img(f"Mosaic output {mosaic.shape}", mosaic)
    
    cv.imwrite("mosaic.png", mosaic)
    
    return mosaic        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n", "--rows",
                        help="Number of rows in tile grid image",
                        type=int,
                        default=8)
    
    parser.add_argument("-l", "--cols",
                        help="Number of columns in tile grid image",
                        type=int,
                        default=16) 
    
    parser.add_argument("-g", "--grid-path",
                        help="Tile grid image path",
                        type=str,
                        default="segs.png")
    
    parser.add_argument("-i", "--image-path",
                        help="Source image path",
                        type=str,
                        default="images/test.jpg")
    
    parser.add_argument("-d", "--debug",
                        help="Show additional parsing and image output",
                        action="store_true",
                        default=False)    
        
    args = parser.parse_args()
    
    tile(**vars(args))



