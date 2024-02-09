import numpy as np
import os
import pickle
import cv2
from debugger import get_camera_view
import glob
import sys
import sympy as sym
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import Pow, re, sqrt, Abs
from scipy import optimize
from helper import fun, computePlane, linear_fit, rodrigues, fitPlane, computePlane, image_creating
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


class LaserCalibration:
    """
    Class for the laser-stripe calibration

    Parameters:
    -----------
    image_path: str
        Path of the folder where are the images in RGB with the chesboard
    cameracalib_name:str
        String of the name of the camera calibration parameters in the pickle format
    lenght_square: float
        The lenght in meters if the size of a square in the chessboard
    corners_shape: tuple of int
        Tuple of the number of corners in the vertical and horizontal
    vertical: boolean
        The position of the laser in the chessboard, of the line is vertical the value is True
    show-results: boolean
        Boolean for showing the pictures with the corners draw in the chessboard
    """

    def __init__(self, images_path, cameracalib_name, length_square = 0.0322, corners_shape = (6, 9), vertical = False, show_results = True):
        
        # Path to the images with the chessboard
        # self.images_path = glob.glob('./mask/1661358793.png') + glob.glob('./mask/1661358848.png') + glob.glob('./mask/1661358973.png')
        self.images_path = os.listdir(images_path)
        # self.images_path = glob.glob('./mask/1661358991.png') + glob.glob('./mask/1661358856.png')
        
        # Checking images path
        self.check_images()

        # Name of the camera calibration parameters
        self.cameracalib_name = cameracalib_name

        # Load camera calibration parameters
        self.mtx, self.dist, self.M, self.N, self.rvecs, self.tvecs, self.newmtx = self.load_pickle(self.cameracalib_name)

        # Calibratio parameters of 1920 x 1080

        # self.mtx = np.array([[1094.64623, 0.0, 902.953571], [0.0, 1092.9396, 508.532186], [0.0, 0.0, 1.0]])
        # self.dist =  np.array([-43.4999723, 1044.32282, -0.00358345845, 0.00610381346, 5.72461047, -43.8878541, 1045.11987, 161.599054, -0.00523922427, 0.000557436342, 0.000153647462, 0.00136077005, 0.000328800805, 0.0413341609])


        # Length of side of calibration square in m
        self.lenght_square = length_square 

        # Corners in the chessboard
        self.corners_shape = corners_shape 

        # configuration options
        self.vertical = vertical # are the lasers vertical or horizontal

        # create empty matrix-> solving for plane equation: ax + by + cz + d = 0 of the laser
        self.Q_c_mat = np.empty((0, 4))

        # number of vertical corners
        self.h = corners_shape[0]

        # number of horizontal corners
        self.w = corners_shape[1]

        # Show result point in the plane ecuation
        self.show_results = show_results

        # Boolean if the chessboard is found
        self.chessboard_found = False

        # Chessboard corners
        self.corners = []

        self.H_c_mat = np.zeros((0, 3))

    def reading_images(self, name):
        '''
        Reading images, undostord them and converting to gray scale

        Parameters:
        -----------
        name: str
            Name of the image that we are using

        Returns:
        --------
        gray: np.array 
            The undistord and denoised image in binary format
        '''

        # load current image
        img_org = cv2.imread('/content/drive/MyDrive/Javii/Datos calibracion laser/images_rectify/' + name , cv2.IMREAD_COLOR)
        # # undistort the image -> use this for finding laser
        # self.img_undist = cv2.undistort(img_org, self.mtx, self.dist) 
        self.img_undist =  img_org
        
        # # denoise image -> for finding checkerboard
        # img_denoised = cv2.fastNlMeansDenoisingColored(
        #     self.img_undist, None, 10, 10, 7, 21)

        # convert to greyscale
        self.gray = cv2.cvtColor(img_org , cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(img_org , cv2.COLOR_BGR2GRAY)

        return gray 

    def check_images(self):
        '''
        Checking if the path of the images are correct, if the path is wrong we will obtain the error
        '''
        # if no images exit
        if (len(self.images_path) == 0):
            print('Error: no photos in images folder')
            sys.exit(0)

    def load_pickle(self, filename):
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
            list_of_objs = pickle.load(doc)
        return list_of_objs


    def obtain_coords(self, image):
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
        x = []
        y = []
        for i, row in enumerate(image):
            for j, column in enumerate(row):
                if 0 not in column:
                    x.append(j)
                    y.append(i)
                    
        return x, y

    def plane_calculation(self, x, points, params): 
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
        dist = 0
        for point in points:
            dist += abs(point[0]* x[0] + point[1]*x[1] + point[2]*x[2] + x[3])

        return dist/len(points)


    def finding_laser(self, a_coord, c_coord, laser_img, gray):
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

        laser_found = False
        for t_candidate in np.linspace(0, 1, num=gray.shape[1]):
            q_candidate = (t_candidate*a_coord+(1-t_candidate)*c_coord).astype(int)
            if (laser_img[q_candidate[1], q_candidate[0]] > 200):
                # TODO: don't just take the first point. take the middle one. laser is still a few pixels wide.
                laser_found = True
                q_coord = q_candidate
                break

        return q_coord, laser_found
        
    def finding_laser_method(self, a_coord, c_coord, laser_img):
        '''
        Function to fin de laser between 2 points in the chessbard

        Parameters:
        -----------
        a_coord: top point of the chessboard
            Point in 2D in the image
        c_coord: bottom point in the chessboard
            Point in 2D in the image
        laser_img: binary mask with one white pixel per column
            Binary mask

        Returns:
        --------
        q_coord: coordinate of the laser point in the image
            2D coordinate in the image
        '''
        laser_found = False
        q_coord = ""
        for t_candidate in np.linspace(0, 1, num=self.img_undist.shape[1]):
            q_candidate = (t_candidate*a_coord+(1-t_candidate)*c_coord).astype(int)
            if (laser_img[q_candidate[1],q_candidate[0]] > 20):

                laser_found = True
                q_coord = q_candidate
                break
        return laser_found, q_coord
    
    def paul_method(self, x_points, y_points, number = 6):

        '''
        Function to obtain the index of the corners that we want

        Returns:
        --------
        top_coord: tuple
            The row and column of the top coord
        mid_coord: tuple
            The row and column of the middle coord
        bot_coord: tuple
            The row and column of the bottom coord
  
        '''
        # OJO pasarle puntos del tablero
        x_coords = np.linspace(np.min(x_points), np.max(x_points), number)

        coordinates = []
        for x_coord in x_coords:
            index = x_points.index(round(x_coord))
            y_coord = y_points[index]
            coordinates.append([x_coord, y_coord])
        
        for coord in coordinates:

            # solve for Q
            Q = np.array([coord[0], coord[1], 0])

            # make Q homog.
            Q = np.append(Q, 1)

            # correspondences are column wise
            # solve PnP to get camera rotation and translation matrix
            # that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame
            ret, rvec, tvec = cv2.solvePnP(
                self.objp, self.corners, self.mtx, np.zeros((5, 1)))  # rvec is rodrigues vector

            # put Q into camera frame using rvec, tvec
            R = rodrigues(rvec)
            T = np.vstack((np.concatenate((R, tvec), axis=1), [0, 0, 0, 1]))

            Q_c = T.dot(Q)
            print('3D point of laser found:')
            print(Q_c[:3])

            # Add Q_c to Q_c_mat
            self.Q_c_mat = np.vstack((self.Q_c_mat, Q_c))
            get_camera_view(rvec, tvec, self.lenght_square)

        print(self.Q_c_mat)
 
 


    def save_pickle(self, name, list_of_objs, directory=None):
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

    def pixel_per_column(self, mask):
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
            p_column= np.where(mask[:,i] == 255)

            # Saving the coordinates of the white pixels and calcualting the average
            points_percolumn.append([i,np.average(p_column)])

        # Returning the coordinates in format ([x, y],[x1, y1])
        return points_percolumn

    def from_pixels_to_mm(self, pixels, plane_param, resolution_x = 0.00032086, resolution_y = 0.00032061):
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
        x = [i[0] for i in pixels]
        y = [i[1] for i in pixels]

        # Converting the x and y pixeles in meters
        y = [i*resolution_y for i in range(len(y))]
        x = [(i)* resolution_x  for i in x]

        # Setting the z axis empty
        Z = []

        # Loop for he coordinates
        for i, pixel in enumerate(pixels):
            # Appending the z value made by the plane equation	
            Z.append((a*pixel[0] + b*pixel[1] + d)/(-c))

        # Return the values for x, y and z
        return x, y, Z

    
    def show_finalresults(self):
        '''
        Plotting the results of the plane equation

        '''
    
        fig = plt.figure()
        Q = self.Q_c_mat.astype(np.double)
        ax = Axes3D(fig)
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def computing_plane(self, ransac = False):
        '''
        Function to compute the [a, b, c, d] plane parameters of the laser plane.
        Parameters:
        -----------
        ransac: boolean
            If you want to use ransac, change the value to True. In other case the last square method it will use

        Returns:
        --------
        l: list
            List of the parameters of the plane equation
        '''
        if (self.Q_c_mat.shape[0] < 3):
            print('Error: not enough laser points found to find plane eqn. Need at least 3')
            sys.exit(0)
        else:
            print('Found a total of ' +
                str(self.Q_c_mat.shape[0]) + ' 3D points. Computing plane.')
            # np.save('Q', Q_c_mat.astype(np.double))

            # solve for the plane
            Q_c_mat = self.Q_c_mat.astype(np.float64)
            
            # RANSAC method
            if ransac:
                l = computePlane(Q_c_mat[:, :3], 10000, 0.001, 0.4, 0.2)
                name = "RANSAC method LAB"
                self.save_pickle(name, l, directory=None)

            # OPTIMIZATION method
            else:
                l = optimize.least_squares(fun, [1,1,1,1], jac='3-point', args=(Q_c_mat[:, :3], [1,1,1,1]), verbose=2, loss='soft_l1', max_nfev=30000, xtol=1e-30, ftol=1e-5)
                # Saving the results
                name = "Last-square_method LAB"
                self.save_pickle(name, l['x'], directory=None)

            if type(l) == int:  # return 0 if RANSAC doesn't pass
                print('Error: maximum RANSAC iterations passed, no solution.')
                sys.exit(0)

        return l
    
    def creating_mask(self, name):
        '''
        Function to create a mask with one white pixel per column

        Parameters:
        -----------
        name: str
            Name of the mask that we want to tranform

        Returns:
        --------
        laser_img: np.array
            Mask with one pixel per column
        imask: np.array
            The original mask
        '''
        # Reading mask of the mask folder
        imask = cv2.imread('/content/drive/MyDrive/Javii/Datos calibracion laser/mask_rectify/' + name)

        # Obtaining coords of the white pixels in the mask
        column, row = self.obtain_coords(imask)

        #Fitting a line based in the white pixels
        m, n = linear_fit(column, row)

        H,W,_ = imask.shape

        # Begin with the points empty
        x_points = []
        y_points = []
        for number in range(W - 1):
            # Creating the one pixel per column with the fit line
            x_points.append(number)
            y_points.append(m*number + n)
            
        # With the pixels create the image
        laser_img = image_creating(x_points, y_points, int(H), int(W))

        return laser_img, imask, x_points, y_points

    def corners_index(self, number=6):
        '''
        Function to obtain the index of the corners that we want

        Returns:
        --------
        top_coord: tuple
            The row and column of the top coord
        mid_coord: tuple
            The row and column of the middle coord
        bot_coord: tuple
            The row and column of the bottom coord
  
        '''
        # The empty lists with the corners
        top_coord, mid_coord, bot_coord = [], [], []

        if not self.vertical:
            if number > self.w:
                print("We have less corners that you want")
            else:
                for corner in np.linspace(0, self.w-1, num = number):
                    # pick 3 corner points in a vertical line in the middle of the checkerboard: 2D a,b,c and 3D A, B, C
                    # a should be top, b middle, c bottom.
                    top_coord.append([0, int(corner)])
                    mid_coord.append([int(self.h/2.0), int(corner)])
                    bot_coord.append([int(self.h-1), int(corner)])
        else:
            if number > self.h:
                print("We have less corners that you want")
            else:
                for corner in np.linspace(0, self.h-1, num = number):
                    # pick 3 corners in a horizontal line in the middle of the checkerboard: 2D a,b,c and 3D A, B, C
                    top_coord.append([int(corner), 0])
                    mid_coord.append([int(corner), int(self.w/2.0)-1])
                    bot_coord.append([int(corner), int(self.w-1)])
        
        return top_coord, mid_coord, bot_coord

    def corner_points(self, top_coord, mid_coord, bot_coord):
        '''
        Function to obtain the pixel corners with the number of corner of the function FindChesscorners

        Parameters:
        -----------
        top_coord: tuple
            The row and column of the top coord
        mid_coord: tuple
            The row and column of the middle coord
        bot_coord: tuple
            The row and column of the bottom coord

        Returns:
        --------
        a_coord: tuple
            Pixel coordinate of the top corner
        b_coord: tuple
            Pixel coordinate of the middle corner
        c_coord: tuple
            Pixel coordinate of the bottom corner
        top_idx: int
            Index of the top corner
        mid_idx: int
            Index of the middle corner
        bot_idx: int
            Index of the bottom corner

        '''
        # get the index of a, b, c
        top_idx = top_coord[0] + top_coord[1]*self.h
        mid_idx = mid_coord[0] + mid_coord[1]*self.h
        bot_idx = bot_coord[0] + bot_coord[1]*self.h
        # Add these print statements to debug
        print(f"top_idx: {top_idx}, mid_idx: {mid_idx}, bot_idx: {bot_idx}")
        print(f"self.corners[top_idx, :, :].shape: {self.corners[top_idx, :, :].shape}")
        print(f"self.corners[mid_idx, :, :].shape: {self.corners[mid_idx, :, :].shape}")
        print(f"self.corners[bot_idx, :, :].shape: {self.corners[bot_idx, :, :].shape}")

        # get the coordinates of a, b, c
        a_coord = self.corners[top_idx, :, :].reshape((2))
        b_coord = self.corners[mid_idx, :, :].reshape((2))
        c_coord = self.corners[bot_idx, :, :].reshape((2))

        return a_coord, b_coord, c_coord, top_idx, mid_idx, bot_idx

    def finding_chessboard_image(self, gray, scale_percent = 20, show = False):
        '''
        Function to find the corners in the chessboard and compute the corners in the camera frame

        Parameters:
        -----------
        gray: np.array
            Image that we want to analyze
        scale_percent: float
            Porcentage of the original image that we want to see
        show: boolean
            If you want to see the corners in the chessboard

        '''
        # Finding corners
        # ret, corners = cv2.findChessboardCorners(gray, self.corners_shape, cv2.CALIB_CB_EXHAUSTIVE, cv2.CALIB_CB_ACCURACY)
        ret, corners = cv2.findChessboardCorners(gray, self.corners_shape)

        # find the corners, findChessboardCorners, use cornerSubPix
        if not ret:
            print("No chessboard pattern found.")
        else:
            # Drawing corners
            f = cv2.drawChessboardCorners(gray, self.corners_shape, corners,
                                True)

            # If you want to see the corners
            if show:
                width = int(self.gray.shape[1] * scale_percent / 100)
                height = int(self.gray.shape[0] * scale_percent / 100)
                dim = (width, height)
                resizedf = cv2.resize(f, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow("Chessboard", resizedf)
                cv2.waitKey(0)
                cv2.destroyWindow("Chessboard")

            # refine corners -> termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # refine corners
            self.corners = cv2.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), criteria)

            # find 3D object points
            self.objp = np.zeros((self.w * self.h, 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:self.h, 0:self.w].T.reshape(-1, 2)

            # scale with length of each square
            self.objp *= self.lenght_square 

        self.chessboard_found = ret

    def ratio(self, a_coord, b_coord, c_coord, q_coord, top_idx, mid_idx, bot_idx, show = False):
        '''
        Function to compute the ratio and geometric relationship

        Parameters:
        -----------
        a_coord: tuple
            Top coord in pixels of the corner
        b_coord: tuple
            Middle coord in pixels of the corner
        c_coord: tuple
            Bottom coord in pixels of the corner
        q_coord: tuple
            Coord in pixels of the laser cross with the vertical corners
        top_idx: int
            Index of the top coord
        mid_idx: int
            Index of the middle coord
        bot_idx: int
            Index of the bottom coord
        show: bool
            True or False if you want to see the points
        
        Return
        -------
        self.sol: tuple
            Coords of the cross between the laser and vertical coords in chessboard frame

        '''
        # convert a,b,c,q to normalized coordinates p_n -> p_u = Kp_n  (p_n = [I|0]p_c), (p_c = [R|t]p_w)
        # so p_n = inv(K)*p_u
        a = np.linalg.inv(self.mtx).dot(np.append(a_coord, 1))
        a = a[:2]/a[2]
        b = np.linalg.inv(self.mtx).dot(np.append(b_coord, 1))
        b = b[:2]/b[2]
        c = np.linalg.inv(self.mtx).dot(np.append(c_coord, 1))
        c = c[:2]/c[2]
        q = np.linalg.inv(self.mtx).dot(np.append(q_coord, 1))
        q = q[:2]/q[2]

        # compute cross ratio using pixels a, b, c
        # ((AB)/(QB))/((AC)/(QC)) = ((ab)/(qb))/((ac)/(qc))
        A = self.objp[top_idx, :]
        B = self.objp[mid_idx, :]
        C = self.objp[bot_idx, :]

        # If you want to see the points in the camera frame
        if show:
            pts = np.vstack((A, B, C))
            plt.scatter(pts[:, 0], pts[:, 1])
            plt.show()
            

        # q is either in between a and b, or between b and c.
        # cross ratio is a line of a, q, b, c or a, b, q, c
        q_left = False
        if np.linalg.norm(a-q) < np.linalg.norm(a-b):
            q_left = True

        # q is between a and b: cr(a,q,b,c) = cr(A,Q,B,C) = (AB/QB)/(AC/QC)
        if q_left:
            ab = np.linalg.linalg.norm(a-b)
            qb = np.linalg.linalg.norm(q-b)
            ac = np.linalg.linalg.norm(a-c)
            qc = np.linalg.linalg.norm(q-c)
            cross_ratio = ((ab)/(qb))/((ac)/(qc))
            AB = np.linalg.norm(A-B)
            AC = np.linalg.norm(A-C)
            BC = AC-AB
            # use cross ratio + geometry
            QB = sym.symbols('QB', real=True)
            QB = sym.solve((AB/QB)/(AC/(BC+QB))-cross_ratio, QB)[0]  # cross
            QC = BC + QB  # geometric relationship: BC + QB = QC
            Qx, Qy = sym.symbols('Qx, Qy')
            self.sol = sym.solve((sqrt(Pow(Qx - B[0], 2) + Pow(Qy - B[1], 2))-QB, sqrt(
                Pow(Qx - C[0], 2) + Pow(Qy - C[1], 2))-QC), (Qx, Qy))

        else:  # cr(a,b,q,c) = cr(A,B,Q,C) = (AQ/BQ)*(AC/BC)
            # q is between b and c... flip the problem (swap a and c from above condition)
            aq = np.linalg.linalg.norm(q-a)
            qb = np.linalg.linalg.norm(q-b)
            ac = np.linalg.linalg.norm(a-c)
            bc = np.linalg.linalg.norm(b-c)

            cross_ratio = ((aq)/(qb))/((ac)/(bc))
            AC = np.linalg.norm(A-C)
            CB = np.linalg.norm(C-B)
            AB = AC-CB
            # use cross ratio + geometry
            QB = sym.symbols('QB', real=True)
            QB = sym.solve(((AB+QB)/QB)/(AC/CB)-cross_ratio, QB)[0]  # cross
            QA = AB + QB  # geometric relationship: BC + QB = QC
            Qx, Qy = sym.symbols('Qx, Qy')
            self.sol = sym.solve((sqrt(Pow(Qx - B[0], 2) + Pow(Qy - B[1], 2))-QB, sqrt(
                Pow(Qx - A[0], 2) + Pow(Qy - A[1], 2))-QA), (Qx, Qy))
        print(self.sol)
        return self.sol


    def visualizing_points(self, a_coord, b_coord, c_coord, q_coord = "", scale_percent = 20):
        '''
        Function to see the a, b and c points

        Parameters:
        -----------
        a_coord: tuple
            Top coord in pixels of the corner
        b_coord: tuple
            Middle coord in pixels of the corner
        c_coord: tuple
            Bottom coord in pixels of the corner
        q_coord: tuple
            Coord in pixels of the laser cross with the vertical corners
        scale_percent: int
            Porcentage of the image that we want resize

        '''
        # visualize the four points a,b,c,q
        pts = np.vstack((a_coord, b_coord, c_coord)).astype(int)
        img_circles = cv2.drawChessboardCorners(
            self.gray, self.corners_shape, self.corners, self.chessboard_found)
  
        # Draw a circle with blue line borders of thickness of 2 px
        for i in range(0, pts.shape[0]):
            # fyi, CVPoint = (x,y) -> x right, y down.
            img_circles = cv2.circle(
                img_circles, (pts[i, 0], pts[i, 1]), 10, (255, 0, 0), 4)

        if q_coord != "":
            img_circles = cv2.circle(
                img_circles, (q_coord[0], q_coord[1]), 10, (255, 0, 0), 4)


        width = int(self.gray.shape[1] * scale_percent / 100)
        height = int(self.gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        resizedf = cv2.resize(img_circles, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow('img', resizedf)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def computing_Q(self):
        '''
        Function to compute Q in the camera frame
        '''
        
        Qx = re(self.sol[0][0])
        Qy = re(self.sol[0][1])

        # solve for Q
        Q = np.array([Qx, Qy, 0])

        # make Q homog.
        Q = np.append(Q, 1)

        # correspondences are column wise
        # solve PnP to get camera rotation and translation matrix
        # that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame
        ret, rvec, tvec = cv2.solvePnP(
            self.objp, self.corners, self.mtx, np.zeros((5, 1)))  # rvec is rodrigues vector

        # put Q into camera frame using rvec, tvec
        R = rodrigues(rvec)
        T = np.vstack((np.concatenate((R, tvec), axis=1), [0, 0, 0, 1]))

        Q_c = T.dot(Q)
        print('3D point of laser found:')
        print(Q_c[:3])

        # Add Q_c to Q_c_mat
        self.Q_c_mat = np.vstack((self.Q_c_mat, Q_c))


    def chessboard_cameraframe(self, points):
        '''
        Function to compute Q in the camera frame
        '''
        total = []
        print(points)
        for coord in points:
          
            x = coord[0]
            y = coord[1]
          

            # solve for Q
            H = list([x, y, 0])

            # make Q homog.
            H = np.append(H, 1)

            # correspondences are column wise
            # solve PnP to get camera rotation and translation matrix
            # that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame
            ret, rvec, tvec = cv2.solvePnP(
                self.objp, self.corners, self.mtx, np.zeros((5, 1)))  # rvec is rodrigues vector

            # put Q into camera frame using rvec, tvec
            R = rodrigues(rvec)
            T = np.vstack((np.concatenate((R, tvec), axis=1), [0, 0, 0, 1]))

            H_c = T.dot(H)
            total.append(H_c)

            # Add h_c to h_c_mat
        self.H_c_mat = np.vstack((self.H_c_mat, H_c[:3]))

    def get_checkerboar_position(self):
 
        x = [self.corners[0][0][0], self.corners[self.corners_shape[0] - 1][0][0], self.corners[(self.corners_shape[0])*(self.corners_shape[1]) - 1][0][0],
            self.corners[(self.corners_shape[0])*(self.corners_shape[1]) - self.corners_shape[0]][0][0]]
        y = [self.corners[0][0][1], self.corners[self.corners_shape[0] - 1][0][1], self.corners[(self.corners_shape[0])*(self.corners_shape[1]) - 1][0][1],
            self.corners[(self.corners_shape[0])*(self.corners_shape[1]) - self.corners_shape[0]][0][1]]
        # z = [self.corners[0][0][2], self.corners[self.corners_shape[0] - 1][0][2], self.corners[(self.corners_shape[0])*(self.corners_shape[1]) - 1][0][2],
        #     self.corners[(self.corners_shape[0])*(self.corners_shape[1]) - self.corners_shape[0]][0][2]]
        return zip(x, y)







