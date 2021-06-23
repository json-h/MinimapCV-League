import numpy as np
import cv2 as cv
import time

w, h = 1920, 1080 # screen res
map_size = 280 # 360 on 1440p

best_methods = ['cv.TM_CCOEFF_NORMED']
champ_img_path = './NunuWillumpSquare.png'
capture = cv.VideoCapture('video/game1.mp4')

template = cv.resize(cv.imread(champ_img_path, 0),(24,24), interpolation = cv.INTER_AREA)
width, height = template.shape

while True:
    isRead, frame = capture.read()
    minimap = frame[h-map_size:h, w-map_size:w]
    screen_gray = cv.cvtColor(minimap, cv.COLOR_BGR2GRAY) # change to grey scale
    screen_copy = screen_gray.copy()

    #Apply template Matching
    res = cv.matchTemplate(screen_copy, template, cv.TM_CCOEFF_NORMED) 
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    print(max_val)

    if (max_val > 0.52):
        
        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv.rectangle(screen_copy, top_left, bottom_right, 255, 2) # draw rect around match
    
    cv.imshow('Matching with cv.TM_CCOEFF', screen_copy)

    if cv.waitKey(25) & 0xFF == ord('q'): # run until user quits
        cv.destroyAllWindows()
        break