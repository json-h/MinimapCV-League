import numpy as np
import numpy.random
import matplotlib.pyplot as plt

def heatmap(name, xcoords, ycoords):
    plt.hist2d(xcoords,ycoords, bins=[np.arange(0,280,3),np.arange(0,280,3)], density=True)
    plt.title(name)
    plt.gca().invert_yaxis()
    plt.show()