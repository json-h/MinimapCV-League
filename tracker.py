import numpy as np
import cv2 as cv
import heatmap as hm

class Champion():
    
    def __init__(self, name):
        self.name = name
        self.xpos = []
        self.ypos = []
        self.image = cv.resize(cv.imread("assets/" + self.name + ".png", cv.IMREAD_UNCHANGED),(23,23), interpolation = cv.INTER_AREA) # read in resized image
        self.template = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY) # convert template into greyscale
        
        # something goes wrong w/ solid colors with the mask, fix this
        """
        self.mask = np.array(cv.split(self.image)[3]) # split template into r,g,b,a channels then create mask array using alpha channel
        """

def prepare_frame(frame):
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(frame, cv.HOUGH_GRADIENT, 1, minDist=2, param1=400, param2=11.4, minRadius=12, maxRadius=14)
    
    if circles is not None:
        circle_mask = np.zeros_like(frame)
        circles = np.uint16(np.around(circles))
        for detected in circles[0,:]:
            center = (detected[0], detected[1])
            cv.circle(circle_mask, center, detected[2], color=(255, 255, 255), thickness=-1)
        return cv.bitwise_and(frame, frame, mask=circle_mask)

def main():

    # todo: find a solution to all of the hardcoded values to support different resolutions
    
    w, h = 1920, 1080 # screen res
    template_w, template_h = 23, 23 # size of template
    map_size = 280 # 360 on 1440p

    capture = cv.VideoCapture('assets/game2speed.mp4')
    
    trackers = [] # array of champions to track
    trackers.append(Champion("Vi"))
    trackers.append(Champion("Volibear"))
    trackers.append(Champion("Kennen"))
    trackers.append(Champion("Sylas"))
    trackers.append(Champion("Shen"))
    trackers.append(Champion("Draven"))
    trackers.append(Champion("Ezreal"))
    trackers.append(Champion("Blitzcrank"))
    trackers.append(Champion("Gwen"))

    while True:
        isRead, frame = capture.read()
        minimap = frame[h-map_size:h, w-map_size:w]
        masked = prepare_frame(minimap)
        if masked is not None:
            for tracked in trackers:
                res = cv.matchTemplate(masked, tracked.template, cv.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                print(max_val)
                if max_val > 0.75:
                    center = [int(max_loc[0] +  template_w/2), int(max_loc[1] + template_h/2)]
                    cv.putText(minimap, tracked.name, [max_loc[0]-5, max_loc[1]-5], cv.FONT_HERSHEY_SIMPLEX, 0.25, (0,255,0), 1)
                    cv.circle(minimap, center, 13, color=(255, 255, 255), thickness=2)
                    tracked.xpos.append(center[0])
                    tracked.ypos.append(center[1])
        cv.imshow('Tracker', minimap)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            for tracked in trackers:
                hm.heatmap(tracked.name, tracked.xpos, tracked.ypos)
            break

if __name__ == "__main__":
    main()