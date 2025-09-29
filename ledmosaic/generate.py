import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import argparse
from .utils import show_img


def rasterize_pts(svg_pts, size_px, viewbox):
    """Convert SVG polygon points to image points
    
    Uses the SVG viewport to convert viewBox units -> CSS pixels.
    Given a size_px of an image (width, height) and the points which 
    are defined with respect to an SVG viewBox, we can map the abstract
    coordinates to screen pixels.
    """
    
    assert type(svg_pts) == list, "You must pass in a list of SVG points"
    
    width_px, height_px = size_px
    minx, miny, width_vb, height_vb = viewbox
    
    img_pts = []
    
    for (x, y) in svg_pts:
        x = round((x - minx)*(width_px / width_vb))
        y = round((y - miny)*(height_px / height_vb))
        img_pts.append((x, y))
        
    return img_pts

def parse_svg():
    """Parse 7-segment digit SVG to extract polygon information"""
    
    tree = ET.parse("digit.svg")
    root = tree.getroot()

    viewbox = [int(elem) for elem in root.attrib['viewBox'].split()]
        
    segs = {child.attrib['id'] : child.attrib['points'] for child in root}

    for seg in segs:
        segs[seg] = segs[seg].split()
        for i in range(len(segs[seg])):
            x, y = segs[seg][i].split(",")
            segs[seg][i] = (int(x), int(y))
            
    return segs, viewbox    

def draw_pts(pts, size, color, separation=0):
    """Create image and draw pixels based on pts list
    
    we add pixel separation between segments otherwise segments
    will be squished together with only a single pixel separation
    the actual separation width is scaled acording to image size
    
    the following formula was derived assuming that 2 pixels is a good
    thickness for a (320, 192) image and 1 pixel is good for (40, 24)
    
    (y2-y1) / (x2-x1) == (2-1) / (320*192 - 40*24) == m == 0.000016534
    y intercept -> y - mx = b, 1 - 40*24*m == 0.98412736
    """
    
    width, height = size
    
    # because openCV expects number of rows, columns, channels
    channels = 3 
    image = np.zeros((height, width, channels), dtype=np.uint8)
    
    pts = np.array(pts, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(image, [pts], color)
    
    if separation == None:
        m = (2-1) / (320*192 - 40*24)
        b = 1 - 40*24*m
        separation = int(m*width*height + b)
        
    cv.polylines(image, [pts], isClosed=True, color=(0, 0, 0), thickness=separation)

    return image




def gen_all_combinations(segs, size, on_color, off_color, debug=False):
    """Generate all 7-segment display combinations
    
    Every output image starts with a base template where all segments are off.
    Then, for every combination of segment on/off, we add the corresponding segments
    to an output image and then return those as an array.
    """
    
    # 7 segments, each one can be on or off
    num_segments = 7 
    num_combinations = 2**num_segments


    width, height = size
    channels = 3 
    imgs = np.zeros((num_combinations, height, width, channels), dtype=np.uint8)

    for i in range(num_combinations):
        for j, seg in enumerate(segs):
            if (1 << j) & i :
                imgs[i, :] += draw_pts(segs[seg]['pts'], size, on_color)
            else:
                imgs[i, :] += draw_pts(segs[seg]['pts'], size, off_color)

        if debug:    
            show_img(f"img {i} ({i:07b})", imgs[i, :])
        
    return imgs

def gen_grid(imgs, n_rows=8, n_cols=16, debug=False):
    """Place all images next to each other to create a n_rows, n_cols grid
    
    If images cannot all be placed in n_rows, n_cols, the resulting image
    is either cut-off or it is extended with a blank background to accomodate
    """
    num_imgs, height, width, channels = imgs.shape
    
    grid = np.zeros((n_rows*height, n_cols*width, channels), dtype=np.uint8)
    
    
    for i in range(n_rows):
        for j in range(n_cols):
            if i*n_cols + j > num_imgs-1:
                continue
            grid[i*height : i*height + height, j*width : j*width + width] = imgs[i*n_cols + j]
            
            if debug:
                show_img(f"grid ({i}, {j})", grid)
    
    return grid
        

def generate(*,
             img_width_px=192,
             img_height_px=320, 
             on_color=(0, 0, 255), 
             off_color=(118, 118, 118),
             debug=False):
    """Use template SVG file to generate all 7-segment display combinations
    
    Args:
        img_width_px : output 7-segment image width
        img_height_px : output 7-segment image height
        on_color : BGR color ordering when segment is on
        off_color : BGR color ordering when segment is off
        debug : show additional parsing output
    """
    
    seg_pts_svg, viewbox = parse_svg()

    if img_width_px < viewbox[-2] or img_height_px < viewbox[-1]:
        print("warning - image size < than viewbox. This causes bad output image artifacts")

    segs = dict()
    
    # convert SVG points to actual image points
    for seg, svg_pts in seg_pts_svg.items():
        img_pts = rasterize_pts(svg_pts, (img_width_px, img_height_px), viewbox)
        segs[seg] = {'pts': img_pts}
    
    # draw these points on the image
    for seg in segs:
        segs[seg]['img'] = draw_pts(segs[seg]['pts'],
                                    (img_width_px, img_height_px),
                                    on_color)
    if debug:
        for seg in segs:
            show_img(seg, segs[seg]['img'])

    # generate all combination of outputs
    imgs = gen_all_combinations(segs,
                                (img_width_px, img_height_px),
                                on_color, off_color, debug)
    
    # create larger image from each smaller combination image
    grid = gen_grid(imgs, debug=debug)
    
    cv.imwrite("segs.png", grid)
    
    return grid
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--img-width-px",
                        help="7-segment output image cols (width) in pixels",
                        type=int,
                        default=11)
    
    parser.add_argument("-r", "--img-height-px",
                        help="7-segment output image rows (height) in pixels",
                        type=int,
                        default=20) 
    
    parser.add_argument("-o", "--on-color",
                        help="7-segment output image color when seg is ON (comma separated BGR ordering)",
                        type=lambda x : tuple(map(int, x.split(","))),
                        default="0,0,255")
     
    parser.add_argument("-f", "--off-color",
                        help="7-segment output image color when seg is OFF (comma separated BGR ordering)",
                        type=lambda x : tuple(map(int, x.split(","))),
                        default="80,80,80")
    
    parser.add_argument("-d", "--debug",
                        help="Show additional parsing and image output",
                        action="store_true",
                        default=False)    
        
    args = parser.parse_args()
    
    generate(**vars(args))