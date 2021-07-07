import numpy as np
import cv2 as cv

class Champion(): # champion to be tracked
    
    def __init__(self, name, role=None):
        self.name = name
        self.role = role # not in use currently
        self.pos = []
        self.image = cv.resize(cv.imread("assets/" + self.name + "Circle.png", cv.IMREAD_UNCHANGED),(23,23), interpolation = cv.INTER_AREA) # read in resized image
        self.template = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY) # convert template into greyscale
        self.mask = np.array(cv.split(self.image)[3]) # split template into r,g,b,a channels then create mask array using alpha channel values

def main():

    # hardcoded values that will be done better at a later date
    w, h = 1920, 1080 # screen res
    template_w, template_h = 23, 23 # size of template
    map_size = 280 # 360 on 1440p

    capture = cv.VideoCapture('assets/game1.mp4')
    
    trackers = [] # array of champions to track
    trackers.append(Champion("Rumble", "Jungle"))
    trackers.append(Champion("Nunu", "Jungle"))

    while True:
        isRead, frame = capture.read()
        minimap = frame[h-map_size:h, w-map_size:w] # cut the minimap out of the video
        screen_gray = cv.cvtColor(minimap, cv.COLOR_BGR2GRAY) # change to grayscale

        for tracked in trackers:
            res = cv.matchTemplate(screen_gray, tracked.template, cv.TM_CCOEFF_NORMED, mask=tracked.mask)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            if (max_val > 0.60):
                center = (int(max_loc[0] +  template_w/2), int(max_loc[1] + template_h/2))
                cv.circle(minimap, center, 13, color=(255, 255, 255), thickness=2)
                tracked.pos.append(center)
        
            cv.imshow('Matching with cv.TM_CCOEFF_NORMED', minimap)

        if cv.waitKey(1) & 0xFF == ord('q'): # run until user quits
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()