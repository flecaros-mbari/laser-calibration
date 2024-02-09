# import modules
import cv2
from laser_calibration import LaserCalibration
import numpy as np
from helper import comparing_images
import os
from tqdm import tqdm

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# print CV version
print('CV Version: ' + cv2.__version__)

# Sizes of the squares of the chessboards
# For the large chessboard fo 31 , 22 we use 3.81 cm 
# For the small one, we use 3.245 cm 

laser_calib = LaserCalibration("mask/*.png", "calibparameters", length_square=0.03245, corners_shape=(9, 6))

# Iterate through images
for fname in tqdm(laser_calib.images_path):
    print("")
    print(fname)

    # Name of the image that we are studing (asuming a name of 10 numbers and 4 for the ".png")
    name = os.path.basename(fname)

    # Reading image and returning the undistord image with one channel (gray)
    gray = laser_calib.reading_images(name)

    # Finding the chessboard in the gray image
    laser_calib.finding_chessboard_image(gray)
    # If the chessboard is not in the image, we continue with the next image
    if laser_calib.chessboard_found:

        # Creating mask of one white pixel per column, returning the mask with one white pixel per column and the original one
        laser_img, imask, x_points, y_points = laser_calib.creating_mask(name)
  
        # You can verify if the mask correspond to the image that you are studing
        if False:
            # The scale parameter is to resized the original image, because if the shape is huge you won't see the complete image.
            comparing_images(laser_img, imask, scale_percent = 20)

        # TODO: generate a list of top,mid,bot coords to do the search on rather than just one point
        # loop through it

        # Finding the index of the corners that we want, 
        # in this case we are looking for vertical corners to obtain the junction of the corners with the laser
        top_coord, mid_coord, bot_coord = laser_calib.corners_index()

        # laser_calib.paul_method(x_points, y_points)

        # Finding the values in pixels of the corners that we want, we know the order of the corners using this link
        # Link of the corners: https://i.stack.imgur.com/I9VXM.jpg
        for coord in range(len(top_coord)):

            a_coord, b_coord, c_coord, top_ind, mid_ind, bot_ind = laser_calib.corner_points(top_coord[coord], mid_coord[coord], bot_coord[coord])

            # find q, the laser point between a,b, and c (corers in the chessboard)
            # search between a and c (top and bottom corners of the vertical line)
            laser_found, q_coord = laser_calib.finding_laser_method(a_coord, c_coord, laser_img)
        
            # laser_calib.visualizing_points(a_coord, b_coord, c_coord, q_coord)

            # The variable laser_found is a boolean, it's true if we found the laser stripe
            print("Laser Found", laser_found)

            # No laser found along corner points line
            if not laser_found:
                print("Failed to find laser point. Skipping picture.")
                continue
            else:
                # laser_calib.visualizing_points(a_coord, b_coord, c_coord, q_coord)
                pass
            
            # If we have the chessboard and the laser found, we can calculate the ratio (following the paper of the README)
            solution_found = laser_calib.ratio(a_coord, b_coord, c_coord, q_coord, top_ind, mid_ind, bot_ind)
            
            # If we don't obtain a solution
            if (len(solution_found) == 0):
                print('No solution found.')
                continue
                
            # If we found a solution, we compute the coordinates of the laser in the camera coord. We save those coordinates
            laser_calib.computing_Q()

        points = laser_calib.get_checkerboar_position()
            
        # Showing the chessboard 
        laser_calib.chessboard_cameraframe(points)


# After we look in all the images for the chessboard and the q_coord of the cross 
# between the laser and the vertical line of the corners
# Finding the 4 parameters of the plane [a, b, c, d] and saving them in a pickle file. 

# l = laser_calib.computing_plane(ransac = False)
# print('Plane equation of laser with optimization [a,b,c,d]: ')
# print(l)
laser_calib.save_pickle("Q_c_LAB", laser_calib.H_c_mat, directory=None)

l = laser_calib.computing_plane(ransac = True)

print('Plane equation of laser with RANSAC [a,b,c,d]: ')
print(l)

# Showing the final results, the points in the plane of the laser
laser_calib.show_finalresults()

fig = plt.figure()
ax = Axes3D(fig)

for xb, yb, zb in laser_calib.H_c_mat:
    ax.plot([xb],[yb],[zb], 'w')
# ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
