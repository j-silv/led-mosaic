import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import argparse


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
        x = (x - minx)*(width_px // width_vb)
        y = (y - miny)*(height_px // height_vb)
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

def draw_pts(pts, size, color, separation=2):
    """Create image and draw pixels based on pts list"""
    
    width, height = size
    
    # because openCV expects number of rows, columns, channels
    channels = 3 
    image = np.zeros((height, width, channels), dtype=np.uint8)
    
    pts = np.array(pts, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(image, [pts], color)
    
    # this adds the separation between segments otherwise segments
    # will be squished together with only a single pixel separation
    cv.polylines(image, [pts], isClosed=True, color=(0, 0, 0), thickness=separation)

    return image

def show_img(title, image):
    """Simple wrapper function around cv.imshow"""
    
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def gen_all_combinations(segs, size, on_color, off_color, debug=False):
    """Generate all 7-segment display combinations
    
    Every output image starts with a base template where all segments are off.
    Then, for every combination of segment on/off, we add the corresponding segments
    to an output image and then return those as an array.
    """

    # all 7-segment images will have this as the base
    template_img = sum(draw_pts(segs[seg]['pts'], size, off_color) for seg in segs)

    if debug:    
        show_img("Template image (all off)", template_img)
        
    # TODO: the rest :)
        

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--img-width-px",
                        help="7-segment output image cols (width) in pixels",
                        type=int,
                        default=192)
    
    parser.add_argument("-r", "--img-height-px",
                        help="7-segment output image rows (height) in pixels",
                        type=int,
                        default=320) 
    
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