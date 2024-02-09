from curses import can_change_color
import numpy as np
import math
import os
import pickle
import cv2
import trimesh
import open3d as o3d
import math

def rodrigues(r):
	# https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf
	theta = np.linalg.norm(r)

	if (theta == 0):
		R = np.identity(3)
	else:
		u = (r/theta).reshape(3)
		R = np.identity(3)*math.cos(theta)+(1-math.cos(theta))*np.outer(u,u)+np.array( [ [0, -u[2], u[1] ] ,[ u[2],0, -u[0] ] ,[ -u[1],u[0],0] ] )*math.sin(theta)
	return R

def fitPlane(x,y,z):
	x_1 = x-np.mean(x)
	y_1 = y-np.mean(y)
	z_1 = z-np.mean(z)

	# Use SVD and take the first two eigenvector (the principal components)
	B = np.stack((x_1,y_1,z_1))
	U, S, VH = np.linalg.svd(B)

	# eigenvectors
	p1 = U[:,0]
	p2 = U[:,1]

	# normal vectors
	p3 = np.cross(p1,p2) # the normal vector of our plane
	p3 = p3/np.linalg.norm(p3)

	# find d from the average point
	d = -(p3[0]*np.mean(x) + p3[1]*np.mean(y) + p3[2]*np.mean(z))
	c = p3
	return c,d


# use RANSAC and PCA to compute plane equation
# A: contains data points [x,y,z]
# max_iters: max number of iterations for RANSAC
# tol: tolerance of loss function
# consensus: fraction of points required for model to be accepted
# inlier: fraction of set for inliers
# returns [a,b,c,d] -> ax + by + cz + d = 0
def computePlane(A,max_iters,tol,consensus,inlier):
	n = A.shape[0]
	x = A[:,0]
	y = A[:,1]
	z = A[:,2]
	counter = 0
	best_fit = 0

	while True:

		if counter > max_iters:
			return best_fit

		# 1. Select a random subset of the original data. Call this subset the hypothetical inliers.
		sample_idx = np.random.choice(n,size=int(n*inlier),replace=False)
		x_s = x[sample_idx]
		y_s = y[sample_idx]
		z_s = z[sample_idx]

		# 2. A model is fitted to the set of hypothetical inliers.
		c,d = fitPlane(x_s,y_s,z_s)

		# 3. All other data are then tested against the fitted model. 
		# Those points that fit the estimated model well, according to some  
		# model-specific loss function, are considered as part of the consensus set.
		# average distance
		# ax + by + cz + d = 0
		l = np.absolute(A.dot(c)+d)
		num_fit = np.sum(l<tol)

		# 4. The estimated model is reasonably good if sufficiently many points 
		# have been classified as part of the consensus set.
		if num_fit > consensus*n:
			best_fit = np.append(c,d) 

		counter += 1

def load_pickle(filename):
    """
	Function to load the pickle files

	Parameters:
	-----------
	filename: str
		Path of the data that we want to load
	
	Returns:
	--------
	list_of_obj:list
		List of the files that we loaded

	"""
    with open(filename, 'rb') as doc:
		# Loading the file in the pickle format
        list_of_objs = pickle.load(doc)
    return list_of_objs

def image_creating(x_total, y_total, H, W): 
  '''
	creating the image with the list of white pixels.

	Parameters:
	-----------
	x_total: list
		Coordinate x of the pixeles.
	y_total: list
		Coordinate y of the pixeles.
	H: int
		Height of the image.
	W: int
		Width of the image. 

	Returns:
	--------
	matrix : np.array 
		The new image with the white pixels inside
  '''
  # Empty matrix at first
  matrix = np.zeros((int(H), int(W)),  dtype=int)

  # For coordinate in coordinate
  for coord in zip(x_total, y_total):
	# If the pixels are in the boundaries
    if coord[1] >= 0 and round(coord[1]) < H and coord[0]>= 0 and coord[0]< W:
		# Change the value of the pixel to white
        matrix[round(coord[1]),round(coord[0])] = 255

  return matrix


def obtain_coords(image):
  '''
  Obtaining coordinates of the white pixeles in a image.

  Parameters:
  -----------
  image: matrix
    Image 

  Returns:
  --------
  x, y: list
    pixels coordenates
  '''
  # Begin with empty values of x and y
  x = []
  y = []
  # For row in the image
  for i, row in enumerate(image):
	# For column in image
    for j, column in enumerate(row):
	  # If the pixel is white, save the coordinates
      if 255 in column:
        x.append(j)
        y.append(i)
        
  return x, y

def linear_fit(x, y):
  '''
  2D points to 3D points

  Parameters:
  -----------
  x: int
    X coordinate in the images of the corners
  y: int
    Y coordinate in the images of the corners


  Returns:
  --------
  m, b: floats
    parameters of the linear fit
  '''
  m, b = np.polyfit(x, y, 1)
  return m, b

def plane_calculation(x, points, params): 
  '''
  Using the points in the chessboard we obtain the error in the plane ecuation

  Parameters:
  -----------
  points: lists of 3D coordenates
    Points
  params: list
    Parameters of the plane

  Returns:
  --------
  dist: float
    Error in the plane ecuation
  '''
  # Begin the error in zero
  dist = 0
  for point in points:
	# Computing the error with a point, with the values of the parameters given
    dist += (abs(point[0]* x[0] + point[1]*x[1] + point[2]*x[2] + x[3]))**2

  return dist/len(points)


def fun(x, points, param):
  '''
  Function to optimize 

  Parameters:
  -----------
  points: lists of 3D coordenates
    Points
  params: list
    Parameters of the plane

  Returns:
  --------
  dist: float
    Error in the plane ecuation
  '''

  dist = plane_calculation(x, points, param)
  
  return dist


# We are not using this function in the code, but the paper used this one
def finding_laser(a_coord, c_coord, laser_img, gray):
	'''
	Function to fin de laser between 2 points in the chessbard

	Parameters:
	-----------
	a_coord: top point of the chessboard
		Point in 3D world coordinate
	c_coord: bottom point in the chessboard
		Point in 3D world coordinate
	laser_img: binary mask with one white pixel per column
		Binary mask
	gray: image to compute the values og height and weight
		Image in RGB

	Returns:
	--------
	q_coord: coordinate of the laser point in the image
		3D coordinate in the image
	laser_found: boolean
		If the laser is found return true, else false
	'''
	# We satar with the laser found in False
	laser_found = False
	for t_candidate in np.linspace(0, 1, num=gray.shape[1]):
		q_candidate = (t_candidate*a_coord+(1-t_candidate)*c_coord).astype(int)
		if (laser_img[q_candidate[1], q_candidate[0]] > 200):
			# TODO: don't just take the first point. take the middle one. laser is still a few pixels wide.
			laser_found = True
			q_coord = q_candidate
			break

	return q_coord, laser_found
	
def finding_laser_method(a_coord, c_coord, laser_img, gray):
	'''
	Function to fin de laser between 2 points in the chessbard

	Parameters:
	-----------
	a_coord: top point of the chessboard
		Point in 3D world coordinate
	c_coord: bottom point in the chessboard
		Point in 3D world coordinate
	laser_img: binary mask with one white pixel per column
		Binary mask

	Returns:
	--------
	x: coordinate x of the laser point in the image
		2D coordinate in the image
	'''
	
	# Coordinates of the a coord
	x_acoord = a_coord[0]
	y_acoord = a_coord[1]

	# Coordinates of the c coord
	x_ccoord = c_coord[0]
	y_ccoord = c_coord[1]
	
	# We start with the laser found in False
	laser_found = False

	# We want to know wich coordinate is bigger
	if x_acoord > x_ccoord:
		top = x_acoord
		bottom = x_ccoord
	else:
		top = x_ccoord
		bottom = x_acoord

	# We dont know if we will find the q_coord, but the default value is zero
	q_coord = 0

	# In the range of the columns of the corner points
	for i in range(round(bottom) , round(top)):

		# Finding where is the white pixel
		y = laser_img[:, i]
		for coord in y:
			if coord == 255 and laser_found != True:
				# We obtain the q value, we substract the H value
				q_coord = 3040 - i
				if y_acoord > y_ccoord:
					max = y_acoord
					min = y_ccoord
				else:
					max = y_ccoord
					min = y_acoord
				
				# if the value is in the boundaries of the corners the laser found is True
				if q_coord < max and q_coord > min:
					laser_found = True
					break

	return laser_found, (i, round(q_coord))

def save_pickle(name, list_of_objs, directory=None):
	'''
	Function to save in pickle format

	Parameters:
	-----------
	name: str
		Name of the file that we want to save
	list_of_objs: list
		List of the parameters that we want to save
	directory: str
		String with the path that we want to save the pickle

	'''
	# By default we have the pickle folder to save the pickle
	if directory == None:
		directory = "pickle/" 
	else:
		subpaths = directory.split("/")
		path = ""
		for sub in subpaths:
			path += sub + "/"

	# If we dont put the name of the file
	if name == "":
		filename = 'data_0.p'
		name_count = 1
		while os.path.exists(filename):
			filename = 'data_{}.p'.format(name_count)
			name_count += 1
	else:
		filename = name + ".p"
	
	# Open and save the file
	with open(directory + filename, "wb") as doc:
		pickle.dump(list_of_objs, doc)

def pixel_per_column(mask):
	'''
	Function to put one white pixel per column in a mask

	Parameters:
	-----------
	mask: image np.array
		The mask that we are using
	Returns:
	--------
	points_percolumn: list
		List of coordinates per column
	'''

	# We starte with a empty points list
	points_percolumn = []
	print(len(mask))
	# Loop for column in the mask
	for i in range(len(mask[0])):

		# Finding where are the white pixels
		p_column= np.where(mask[:,i] > 0)

		# Saving the coordinates of the white pixels and calcualting the average
		points_percolumn.append([i,np.average(p_column)])

	# Returning the coordinates in format ([x, y],[x1, y1])
	return points_percolumn

def pixel_per_column_gt(mask):
	'''
	Function to put one white pixel per column in a mask

	Parameters:
	-----------
	mask: image np.array
		The mask that we are using
	Returns:
	--------
	points_percolumn: list
		List of coordinates per column
	'''

	# We starte with a empty points list
	points_percolumn = []

	# Loop for column in the mask
	for i in range(len(mask[0])):

		# Finding where are the white pixels
		p_column= np.where(mask[:,i] > 0)

		# Saving the coordinates of the white pixels and calcualting the average
		points_percolumn.append([i,np.average(p_column)])

	# Returning the coordinates in format ([x, y],[x1, y1])
	return points_percolumn

def from_pixels_to_mm(pixels, plane_param, resolution = 0.00041):
	'''
	Function to compute the X, Y and Z coordinate in the camera frame based in the 
	plane equation and the reoslution of the image

	Parameters:
	-----------
	pixels: list
		The mask that we are using
	plane_param: list
		The list of the a, b, c and d paramaeters of the laser calibration
	resolution: float
		The resolution of one pixel in the image

	Returns:
	--------
	x, y, Z: lists
		Lists of the coordinates in teh camera frame
	'''

	# Reading the parameters
	a, b, c, d = plane_param

	# Reading the x and y points
	mtx, dist, M, N, rvecs, tvecs, newmtx = load_pickle("pickle/" + "calibparameters.p")
	x = [(i[0]- mtx[0][2])/mtx[0][0] for i in pixels]
	y = [(i[1]- mtx[1][2])/mtx[1][1]  for i in pixels]

	# Converting the x and y pixeles in meters
	# y = [i*resolution  for i in y]
	# x = [(i)* resolution  for i in x]

	# Setting the axis empty
	X = []
	Y = []
	Z = []

	# Loop for he coordinates
	for i, pixel in enumerate(pixels):
		if math.isnan(pixel[0]) == False and math.isnan(pixel[1]) == False:
			X.append(x[i]*-d/(a*x[i] + b*y[i] + c))
			Y.append(y[i]*-d/(a*x[i] + b*y[i] + c))
			Z.append(-d/(a*x[i] + b*y[i] + c))


	# Return the values for x, y and z
	return X, Y, Z


def comparing_images(image1, image2, scale_percent = 20):
	'''
	Comparing and searching if the images are correct

	Parameters:
	-----------
	image1: np.array
		First image to compare
	image2: np.array
		Second image to compare
	scale_percent: int
		The porcentage of the image to resize

	'''
	# Obtaining the shape of the images and rescaling them
	width = int(image1.shape[1] * scale_percent / 100)
	height = int(image1.shape[0] * scale_percent / 100)
	dim = (width, height)

	# Combining the images (half and half)
	combine = cv2.addWeighted(image1.astype(np.uint8),0.5,cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.uint8) ,0.5,0)

	# Computing the similarity in the images
	OR = cv2.bitwise_and(image1.astype(np.uint8), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.uint8) )

	# Resizing the images
	resized = cv2.resize(combine, dim, interpolation = cv2.INTER_AREA)
	resized1 = cv2.resize(OR, dim, interpolation = cv2.INTER_AREA)

	# Showing the images
	cv2.imshow("Mask",  resized)
	cv2.imshow("OR", resized1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


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

def make_point_cloud(points, folder):

	points = np.asarray(points)
	colors = np.ones((len(points), 3))
	colors = colors*150
	points_and_color = np.c_[points, colors]
	path = folder + ".ply"
	create_ply(points_and_color, path, len(points_and_color))

def do_mesh(nombre_archivo, nombre_para_mesh):
	
	print("Reading point cloud")
	pcd = o3d.io.read_point_cloud(nombre_archivo)
	print(pcd)
	pcd.estimate_normals()

	# estimamos radio para rolling bal
	distances = pcd.compute_nearest_neighbor_distance()

	avg_dist = np.mean(distances)
	radius = 100 * avg_dist
	# Rolling ball

	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
			pcd,
			o3d.utility.DoubleVector([radius, radius * 2]))

	print("Creating vertex and triangles")
	vertex = np.asarray(mesh.vertices)
	triangles = np.asarray(mesh.triangles)
	normals = np.asarray(mesh.vertex_normals)
	tri_mesh = trimesh.Trimesh(vertex, triangles, vertex_normals=normals)
	print("Exporting mesh")
	tri_mesh.export(nombre_para_mesh)


def obtener_ply(data):
	path_mesh = data + ".stl"
	name = data + ".ply"
	path = os.path.join("ply/", name)



