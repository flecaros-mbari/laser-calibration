# Based in the link: https://stackoverflow.com/questions/66866952/open3d-compute-distance-between-mesh-and-point-cloud
import copy
import pandas as pd
import time
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from plyfile import PlyData, PlyElement
from mayavi import mlab
from helper import create_ply
import os


class Compare:
    """
    Class for compare point clouds

    Parameters:
    -----------
    comparing_clouds: boolean
        Boolean wich represent if the meshes are point clouds or stl-obj
    path_1: str
        Path to the first point cloud
    path_2: str
        Path to the second point cloud

    """
    def __init__(self, comparing_clouds, path_1, path_2):
        '''
        Function to compute the ratio and geometric relationship

        Parameters:
        -----------
        comparing_clouds: boolean
            Boolean wich represent if the meshes are point clouds or stl-obj
        path_1: str
            Path to the first point cloud
        path_2: str
            Path to the second point cloud
        '''

        # Threshold of te feature algorithm
        self.threshold = 0.02

        # Boolean of the point clouds (True of ply-ply and False other case)
        self.comparing_clouds = comparing_clouds

        # Boolean variable, True if we have 2 points clouds and False if we are comparing a mesh and a point cloud
        self.clouds = False
        if comparing_clouds == True:
            self.clouds = True
            self.reading_clouds(path_1, path_2)
        else:
            self.reading_cloud_mesh(path_1, path_2)

        # Inicial Homograpy matrix to iterate with IPC and features
        self.trans_init = np.array([[0.99999362875, 0.00325179925,  0.001227609146,  -0.01416855235], 
                                    [-0.0032553886075,  0.99999088925,  0.002497136185 ,  -8.8438727E-5],
                                    [-0.00121976050475,  -0.002501135865, 0.99999539325 ,  0.00874441261],
                                    [0.00000000, 0.00000000, 0.00000000, 1.00000000]])

    def reading_clouds(self, path1, path2):
        '''
        Function to read the point clouds

        Parameters:
        -----------
        path_1: str
            Path to the first point cloud
        path_2: str
            Path to the second point cloud

        '''
        # Reading the first point cloud
        self.cloud_1 = o3d.io.read_point_cloud(path1)

        # Reading the second point cloud
        self.cloud_2 = o3d.io.read_point_cloud(path2)

    def reading_cloud_mesh(self, path_cloud, path_mesh):
        '''
        Function to read the point clouds and meshes

        Parameters:
        -----------
        path_1: str
            Path to the first point cloud
        path_2: str
            Path to the second mesh
        '''
        # Reading the point cloud
        self.cloud_1 = o3d.io.read_point_cloud(path_cloud)

        # Reading the mesh
        self.mesh = o3d.io.read_triangle_mesh(path_mesh)

        # Creating a empty geometry
        self.cloud_2 = o3d.geometry.PointCloud()  

        # Transform the mesh into a point cloud
        self.cloud_2.points = self.mesh.vertices

    def visualization_double(self):
        '''
        Function to visualize the meshes

        Parameters:
        -----------
        path_1: str
            Path to the first point cloud
        path_2: str
            Path to the second mesh
        '''
        if self.comparing_clouds == True:

            self.cloud_1.paint_uniform_color([0, 0, 1])
            self.cloud_2.paint_uniform_color([0.5, 0.5, 0])
            o3d.visualization.draw_geometries([self.cloud_1, self.cloud_2])
        else:
            self.cloud_1.paint_uniform_color([0, 0, 1])
            self.mesh.paint_uniform_color([0.5, 0.5, 0])
            o3d.visualization.draw_geometries([self.cloud_1, self.mesh])

    def visualization_simple(self, mesh1):
        '''
        Function to visualize the mesh

        Parameters:
        -----------
        path_1: str
            Path to the mesh
        '''

        # Changing color of the mesh
        mesh1.paint_uniform_color([0, 0, 1])

        # Seeing the mesh
        o3d.visualization.draw_geometries([mesh1])

    def calculate_distances(self):
        '''
        Function to compute the distances of the 2 point clouds and show some histographs of the results
        '''

        def write_ply_with_color(coordinates, colors):
            # Crear lista de vértices con coordenadas y colores
            vertex_data = []
            for i in range(len(coordinates)):
                vertex_data.append((coordinates[i][0], coordinates[i][1], coordinates[i][2]))

            make_point_cloud(vertex_data, colors)

        def make_point_cloud(points, color):

            points = np.asarray(points)
        
            points_and_color = np.c_[points, color]
            print(points_and_color )
        
            create_ply(points_and_color,  len(points_and_color))

        def create_ply(vertices, size):
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
            with open("ply/color.ply", 'w') as f:
                f.write(ply_header % dict(vert_num = size))
                np.savetxt(f, vertices, '%f %f %f %f %f %f')
                # Calculate distances of pc_1 to pc_2.

        # We use the cloud of the mesh that we create before, becuase the function doesn't allow the meshes
        dist_pc1_pc2 = self.cloud_1.compute_point_cloud_distance(
            self.cloud_2)

        # Dist_pc1_pc2 is an Open3d object, we need to convert it to a numpy array to
        # acess the data
        dist_pc1_pc2 = np.array(dist_pc1_pc2)

        coordinates_cloud1 = np.asarray(self.cloud_1.points)

        import plotly.graph_objects as go



        fig = go.Figure()

        # Add the 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=coordinates_cloud1[:, 0],
            y=coordinates_cloud1[:, 1],
            z=coordinates_cloud1[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=dist_pc1_pc2,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Distance')
            )
        ))

        # Set the camera angle and distance
        fig.update_layout(scene_camera=dict(
            eye=dict(x=1.7, y=-0.7, z=0.5),
            center=dict(x=0, y=0, z=0)
        ))

        # Set the layout to show the colorbar on the side of the plot
        fig.update_layout(coloraxis_colorbar=dict(
            title='Distance',
            yanchor='middle',
            y=0.5,
            len=0.75,
            thickness=20,
            tickfont=dict(size=10)
        ))

        # Set the layout margins to make room for the colorbar
        fig.update_layout(margin=dict(
            l=0, r=0, b=0, t=0
        ))

        # Show the figure
        fig.show()
 



        # # Crear figura
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(111, projection='3d')
        # color_map = plt.cm.get_cmap('RdYlGn')

        # # Gráfico de dispersión 3D con colores mapeados
        # scatter = ax.scatter(coordinates_cloud1[:, 0], coordinates_cloud1[:, 1], coordinates_cloud1[:, 2], c= -dist_pc1_pc2 , cmap=color_map)

        # # Ajustes
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # cbar = plt.colorbar(scatter)
        # cbar.ax.set_ylabel('Values')

        # plt.show()
        write_ply_with_color( coordinates_cloud1, color_map(dist_pc1_pc2)[:, :3])
        # Let's make a boxplot, histogram and serie to visualize it.
        # We'll use matplotlib + pandas.

        # # Transform to a dataframe
        # df = pd.DataFrame({"distances": dist_pc1_pc2})

        # # Some graphs
        # ax1 = df.boxplot(return_type="axes")  # BOXPLOT
        # ax2 = df.plot(kind="hist", alpha=0.5, bins=1000)  # HISTOGRAM
        # ax3 = df.plot(kind="line")  # SERIE
        # plt.show()


    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." %
              distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def draw_registration_result(self, source, target, transformation):
        '''
        Function to show the meshes with a transformation

        Parameters:
        -----------
        sourse: point cloud
            Point cloud
        target: point cloud
            Point cloud that is the target
        transformation: matrix
            Homography matrix to move the first point cloud to the target
        '''

        # Copy of the point clouds
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        # Changing color
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

        # Using the tranformation matrix
        source_temp.transform(transformation)

        # Seeing the geomtries
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                          zoom=0.4459,
                                          front=[0.9288, -0.2951, -0.2242],
                                          lookat=[0, 0, 0],
                                          up=[-0.3402, -0.9189, -0.1996])

    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def prepare_dataset(self, voxel_size):
        print(":: Load two point clouds and disturb initial pose.")

        # demo_icp_pcds = o3d.data.DemoICPPointClouds()
        self.cloud_1.transform(self.trans_init)
        self.draw_registration_result(
            self.cloud_1, self.cloud_2, np.identity(4))

        source_down, source_fpfh = self.preprocess_point_cloud(
            self.cloud_1, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(
            self.cloud_2, voxel_size)
        return self.cloud_1, self.cloud_2, source_down, target_down, source_fpfh, target_fpfh

    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)

        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return result

    def IPC(self, pc_1, pc_2):

        # if self.comparing_clouds == True:
        # Load a previos transformation to register pc_2 on pc_1
        # I finded it with the Fast Global Registration algorithm, in Open3D

        self.draw_registration_result(pc_1, pc_2, self.trans_init)

        print("Initial alignment")
        evaluation = o3d.pipelines.registration.evaluate_registration(
            pc_1, pc_2, self.threshold, self.trans_init)
        # better the best
        print(evaluation)
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pc_1, pc_2, self.threshold, self.trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     pc_1, pc_2, self.threshold, self.trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        self.draw_registration_result(pc_1, pc_2, reg_p2p.transformation)

    def features(self):
        print("FEATURES")
        # FEatures
        voxel_size = 0.05  # means 5cm for this dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(
            voxel_size)

        start = time.time()
        result_ransac = self.execute_global_registration(source_down, target_down,
                                                         source_fpfh, target_fpfh,
                                                         voxel_size)
        print("Global registration took %.3f sec.\n" % (time.time() - start))
        print(result_ransac)
        self.draw_registration_result(
            source_down, target_down, result_ransac.transformation)

        result_icp = self.refine_registration(source, target, source_fpfh, target_fpfh,
                                              voxel_size, result_ransac)
        print(result_icp)
        self.draw_registration_result(
            source, target, result_icp.transformation)

        return result_ransac


if __name__ == '__main__':

    # This is the main of the code, here you can compare meshes with point clouds, 
    # I tried to use the meshes in obj or stl but the algorithm doesen't work well

    # Begin the class, the first parameter is True if the 2 meshes are ply, in other case is False
    comparing = Compare(True, "ply/GT LAB.ply", "ply/LAB pred illu.ply")

    # Visualization of the point cloud
    comparing.visualization_double()

    # Calculating distances
    comparing.calculate_distances()

    # Computing the IPC and showing the results
    comparing.IPC(comparing.cloud_1, comparing.cloud_2)

    # # Computing with features and showing the results
    comparing.features()


