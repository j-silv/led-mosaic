import numpy as np
import cv2 as cv 
from .tile import tile
 

def video(video_path=2):
    """Run mosaic tiling on video input"""
    
    cap = cv.VideoCapture(video_path, cv.CAP_V4L)
    
    ret = cap.set(cv.CAP_PROP_FRAME_WIDTH,1280.0)
    ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT,1024.0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        mosaic = tile(image_data=frame)
        
        # Display the resulting frame
        cv.imshow('original', frame)
        cv.imshow('mosaic', mosaic)
        
        if cv.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    video()