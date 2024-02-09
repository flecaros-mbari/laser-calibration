import rlcompleter
from helper import  load_pickle, from_pixels_to_mm, pixel_per_column, make_point_cloud, do_mesh, obtener_ply, pixel_per_column_gt
import glob
import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm


plot = False

def calculate_angle(param):
  """
  Function to obtain the angles of the plane and the distances of the plane from the origin of the camera
  
  param: list
  param: list of the 4 parameters of the plane

  return
  anglex: angle of the plane with the x axis
  angley: angle of the plane with the y axis
  anglez: angle of the plane with the z axis
  x_value: distance from the plane crossing the x axis to the 0
  y_value: distance from the plane crossing the y axis to the 0
  z_value: distance from the plane crossing the z axis to the 0

  """

  # vector = np.array([1, 1, -param[2]/(param[0] + param[1])])
  vectorA = np.array([0, 1, -param[2]/(param[0]*0 + param[1])])
  vectorB = np.array([5, 2, -param[2]/(param[0]*5 + param[1]*2)])
  vector = vectorB - vectorA
  vector = [param[0], param[1], param[2]]

  # Asuming the other coordinates cero when we are looking for the intersection
  x_value = -param[3]/ param[0]
  y_value = -param[3]/ param[1]
  z_value = -param[3]/ param[2]

  # Creating the vectors
  v_x = np.array([x_value, 0, 0])
  v_y = np.array([0 , -y_value, 0])
  v_z = np.array([0, 0, z_value])

  # Using the equation of the angles between to planes
  anglex = np.arccos((vector[0]*v_x[0]+ vector[1]*v_x[1] + vector[2]*v_x[2])/(norm(v_x)* norm(vector)))
  angley = np.arccos((vector[0]*v_y[0]+ vector[1]*v_y[1] + vector[2]*v_y[2])/(norm(v_y)* norm(vector)))
  anglez = np.arccos((vector[0]*v_z[0]+ vector[1]*v_z[1] + vector[2]*v_z[2])/(norm(v_z)* norm(vector)))

  return math.degrees(anglex), math.degrees(angley), math.degrees(anglez), x_value, y_value, z_value


def comparing_images(img1, path, image):

  """
  Function to see the differences in the z axis from two different images
  
  Input
  img1: string
  img1: first image folder  
  img2: string
  img2: second image folder
  image: name of the image

  Return
  np.sqrt(rmse)/len(Z1): RMSE of the two images
  X2: Values of the x coordinate
  value: List of the values of the RMSE**2

  """
  path_img, ext = os.path.splitext(os.path.basename(image))
  img1 = cv2.imread(img1 + path_img+ ext, 0)
  img2 = cv2.imread(path + path_img+ ext, 0)

  # Obtaining coordinates of the pixels of the images
  points1 = pixel_per_column(img1)
  points2 = pixel_per_column(img2)


  points11=[]
  points22 =[]

  for number in range(len(points1)):

    # If we are comparing point with point (not considering the nan's)
    if math.isnan(points1[number][1]) == False and math.isnan(points2[number][1]) == False:
      # print(points1[number], points2[number])
      points11.append(points1[number])
      points22.append(points2[number])

  # Setting the metrics
  average = []
  rmse = 0

  # Transforming pixels to 3D points
  X1, Y1, Z1 = from_pixels_to_mm(points11, plane_param)
  X2, Y2, Z2 = from_pixels_to_mm(points22, plane_param)

  # Value of the differences between the coordinates
  value = []
  for num, elemento in enumerate(Z1):

    rmse += (abs(elemento-Z2[num])**2)
    average.append(abs(elemento-Z2[num]))
    value.append(abs(elemento-Z2[num])**2)

  return np.sqrt(rmse)/len(Z1), X2, value

def mapping_pointcloud(images):
  """
  Function who takes a lot of images and a movement distance between them and creates a point cloud

  Input
  images: list
  images: List of the images that where taken with the same distance between them
  """

  # Sorting by name
  images.sort()

  # Empty points
  X_total = []
  Y_total = []
  Z_total = []


  # # # Loop for the images
  for i, image in enumerate(images):

    # Reading mask (distorted) 
    imag = cv2.imread(image)
    back = np.zeros(imag.shape)

    # Undistord the mask
    img_undist = cv2.undistort(imag, mtx, dist)

    # Using only the center of the images to reduce the noise, it will chamge with different sizes of images
    back[400:700, 495:1335] = img_undist[400:700, 495:1335]

    # Transforming the mask in a pixel per column
    points = pixel_per_column(back)

    # Computing the coordinates
    X, Y, Z = from_pixels_to_mm(points, plane_param)

    # Setting the movement in meters between each step
    movement = 0.001

    # Caluclating the Y value with the movement (in this case was vertical movement)
    Y = [j + i*movement for j in Y]

    # Append the coordinates 
    X_total = np.concatenate((X_total, X), axis=None)
    Y_total = np.concatenate((Y_total, Y), axis=None)
    Z_total = np.concatenate((Z_total, Z), axis=None)
    # print( np.max(Z)- np.min(Z))

  # # Tranforming points
  points1 = np.c_[X_total, Y_total, Z_total]

  print("Making point cloud")
  name = "LAB pred bright"

  # Creating point cloud
  make_point_cloud(points1 , name)


def create_ply(vertices, filename, size):
	# Creating the folder ply
    if not os.path.isdir("ply/"):
        os.mkdir("ply/")

    # The header of the file
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    # We create the file with the information of: (x, y, z)
    with open("ply/" + filename, 'w') as f:
        f.write(ply_header % dict(vert_num = size))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')
  
# I have to compute the x and y of the camera view and then Z

# Methods to compute the laser calibration (LASt Square method optimization or RANSAC)
# name = "Last-square_method"
name = "RANSAC method LAB"

# Load optimization results
plane_param = [0.00498, - 0.93126, 0.36431, - 0.24459]
print("Parameters", plane_param)
print(calculate_angle(plane_param))


# # Load calibration parameters
# mtx, dist, M, N, rvecs, tvecs, newmtx = load_pickle("pickle/" + "FERcalibparameters.p")
# print(mtx)

## Comparing in 3D, reading images

mask_bright = sorted(glob.glob("/home/fernanda/Desktop/light/mask/*.png"))
threshold_bright = sorted(glob.glob("/home/fernanda/Desktop/light/threshold/*.png"))
net_bright = sorted(glob.glob("/home/fernanda/Desktop/light/predicted_0.15/*.png"))

print(len(mask_bright[0]), len(threshold_bright), len(net_bright))


rmse_thr = []
rmse_net = []

for num, image in enumerate(net_bright):
  rmse_thr.append(comparing_images("/home/fernanda/Desktop/light/mask/", "/home/fernanda/Desktop/light/threshold/", image)[0])
  rmse_net.append(comparing_images("/home/fernanda/Desktop/light/mask/", "/home/fernanda/Desktop/light/predicted_0.15/", image)[0])
  
  # Plotting

  # fig, ax = plt.subplots(figsize=(12, 8))
  # columns = ["thr", "net"]
  # plt.xticks([1 , 2], ["thr", "net"], fontsize=25)
  # plt.ylabel("Error [meters]", fontsize=20)
  # plt.xlabel("X coordinate [meters]", fontsize=20)
  # plt.title("Illuminated dataset: 3D Reconstruction error with proposed method segmentation", fontsize=20)
  # ax.scatter(comparing_images("/home/fernanda/Desktop/light/mask/", "/home/fernanda/Desktop/light/predicted_0.15/", image)[1],comparing_images("/home/fernanda/Desktop/light/mask/", "/home/fernanda/Desktop/light/predicted_0.15/", image)[2])
  # plt.show()

  plt.title("Illuminated dataset: 3D Reconstruction error with threshold segmentation", fontsize=20)
  plt.ylabel("Error [meters]", fontsize=20)
  plt.xlabel("X coordinate [meters]", fontsize=20)
  ax.scatter(comparing_images("/home/fernanda/Desktop/light/mask/", "/home/fernanda/Desktop/light/threshold/", image)[1],comparing_images("/home/fernanda/Desktop/light/mask/", "/home/fernanda/Desktop/light/threshold/", image)[2])
  plt.show()

print("RMSE NET BRIGHT", np.average(rmse_net))
print("RMSE THR BRIGHT", np.average(rmse_thr))

# If you want to plot something
if plot:

 # Plotting the results
 fig = plt.figure()
 ax1 = fig.add_subplot((111),  projection='3d')

  # 3D plot
 ax1.scatter(X_total, Y_total, Z_total, cmap='viridis', c = Z_total, marker='.')
 ax1.scatter(points1[0], points1[1], points1[2], cmap='viridis', c = points1[2], marker='.')
 ax1.scatter(points2[0], points2[1], points2[2], cmap='viridis', c = points2[2], marker='.')
 # Setting the limits of the axis
 ax1.set_xlim3d(0,1)
 ax1.set_ylim3d(-0.5,0.5)
 ax1.set_zlim3d(-1,0)

 # Name the labels
 ax1.set_xlabel('$X$ [m]')
 ax1.set_ylabel('$Y$ [m]')

 # Set the title
 ax1.set_title("3D reconstruction")
 ax1.yaxis._axinfo['label']['space_factor'] = 3.0
 ax1.set_zlabel('$Z$ [m]', fontsize=15, rotation = 0)

 # Showing the 3D plot
 plt.show()
