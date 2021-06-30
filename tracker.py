import numpy as np
import cv2 as cv

def create_template(path):
    
    template = cv.resize(cv.imread(path, cv.IMREAD_UNCHANGED),(23,23), interpolation = cv.INTER_AREA)
    
    channels = cv.split(template) # split image into r,g,b,alpha channels
    mask = np.array(channels[3]) # create mask array using alpha channel values

    return template, mask

def main():
    w, h = 1920, 1080 # screen res
    map_size = 280 # 360 on 1440p

    template_path = 'assets/RumbleCircle.png'
    capture = cv.VideoCapture('assets/game1.mp4')
    
    init = create_template(template_path)
    template = cv.cvtColor(init[0], cv.COLOR_BGR2GRAY)
    mask = init[1]

    width, height = template.shape
    pos = []

    while True:
        isRead, frame = capture.read()
        minimap = frame[h-map_size:h, w-map_size:w]
        screen_gray = cv.cvtColor(minimap, cv.COLOR_BGR2GRAY) # change to grayscale

        res = cv.matchTemplate(screen_gray, template, cv.TM_CCOEFF_NORMED, mask=mask)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if (max_val > 0.60):
            
            print(max_val)

            center = (int(max_loc[0] + width/2), int(max_loc[1] + height/2))
            cv.circle(minimap, center, 13, color=(255, 255, 255), thickness=2)
            pos.append(center)
        
        cv.imshow('Matching with cv.TM_CCOEFF_NORMED', minimap)

        if cv.waitKey(1) & 0xFF == ord('q'): # run until user quits
            cv.destroyAllWindows()
            
            isRead, frame = capture.read()
            minimap = frame[h-map_size:h, w-map_size:w]

            for detected in range(len(pos)-1):
                if abs(pos[detected][0]-pos[detected+1][0]) < 50 or abs(pos[detected][1]-pos[detected+1][1]) < 50:
                    cv.line(minimap,pos[detected],pos[detected+1],(0,255,0),1)
            
            cv.imshow('Matching with cv.TM_CCOEFF_NORMED', minimap)

            if cv.waitKey(100000) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break

if __name__ == "__main__":
    main()