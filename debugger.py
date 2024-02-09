import numpy as np 
from helper import rodrigues
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def chessboard_cameraframe(self):
    '''
    Function to compute Q in the camera frame
    '''

    for coord in self.corners:
        x = coord[0][0]
        y = coord[0][1]

        # solve for Q
        H = np.array([x, y, 0])

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

        # Add h_c to h_c_mat
        self.H_c_mat = np.vstack((self.H_c_mat, H_c))
    
    fig = plt.figure()
    Q = self.H_c_mat.astype(np.double)
    ax = Axes3D(fig)
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_camera_views(img_points, world_points, mtx, dist):
    x_checker, y_checker, z_checker = get_checkerboar_position()
    fig = plt.figure()
    ax = Axes3D(fig)
    verts = [list(zip(x_checker, y_checker, z_checker))]
    ax.add_collection3d(Poly3DCollection(verts))
    count = 0
    for i in range(len(img_points)):
        # Find the rotation and translation vectors.
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(world_points[i], img_points[i], mtx, dist)
        x, y, z = get_camera_view(rvec, tvec)
        camera_verts = [list(zip(x, y, z))]
        cam = Poly3DCollection(camera_verts)
        color = np.random.rand(3)
        cam.set_color(colors.rgb2hex(color))
        cam.set_edgecolor('k')
        ax.text(x[0]-1, y[0]-1, z[0], str(count), color=colors.rgb2hex(color))
        ax.add_collection3d(cam)
        count += 1
        
    # Xb, Yb, Zb = cubic_plot() # to get the same unit length in the three axis
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    plt.title('Camera views of the checkerboard.')
    plt.show()
    cv2.destroyAllWindows()

    
def get_camera_view(rvec, tvec, sq_size):
    T = np.zeros((4, 4), dtype=np.float64)
    verts = np.zeros((4, 3), dtype=np.float64)
    rot, _ = cv2.Rodrigues(rvec)
    world_rot = np.transpose(rot)
    world_p = -world_rot.dot(tvec)
    T[0:3, 0:3] = world_rot
    T[0:3, 3] = world_p[:, 0]
    T[3, 3] = 1
    cam_side = sq_size
    p1 = T.dot(np.array([0, 0, 0, 1])) # transformation of the camera 4 corners
    p2 = T.dot(np.array([cam_side, 0, 0, 1]))
    p3 = T.dot(np.array([cam_side, cam_side, 0, 1]))
    p4 = T.dot(np.array([0, cam_side, 0, 1]))
    verts[0, :] = p1[0:3]; verts[1, :] = p2[0:3]; verts[2, :] = p3[0:3]; verts[3, :] = p4[0:3]
    local_z_min = np.min(verts[:, 2])
    # if local_z_min < self.z_view:
    #     self.z_view = local_z_min
    return verts[:, 0], verts[:, 1], verts[:, 2]
