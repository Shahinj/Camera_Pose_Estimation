import numpy as np
from imageio import imread
from mat4py import loadmat
from cross_junctions import cross_junctions

# Load the boundary.
bpoly = np.array(loadmat("bounds.mat")["bpolyh1"])

# Load the world points.
Wpts = np.array(loadmat("world_pts.mat")["world_pts"])

# Load the example target image.
I = imread("example_target.png")

Ipts = cross_junctions(I, bpoly, Wpts)

# You can plot the points to check!
print(Ipts)

# 
# 
# def plot(I):
#     import matplotlib.pyplot as plt
#     plt.imshow(I, cmap = 'gray')
#     plt.show()
# 
# 
# for i,row in enumerate(Ipts.T):
#     x,y = row
#     x,y = int(x), int(y)
#     I[y,x] = 255
#     
# plot(I)