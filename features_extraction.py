# -------------------------Import Libraries------------------------------#
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
import os
from sympy import Point3D, Plane
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3
import copy

# -------------------------Set initial values------------------------------#
Threshold1 = [3, 9.333]
initial = 1
initial1 = 1
numberofthre = len(Threshold1)

# Prompting user to select the parent folder
path = askdirectory(title='Select Folder')

# Loops through the subfolders in the parent folder and finds the EDM-1.csv files
for subdir, dirs, files in os.walk(path):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith("EDM-1.csv"):
            datafile = pd.read_csv(filepath, header=None)
            # Delete reference coordinate and deviation columns, keep test coordinate columns
            data3D = datafile.drop(datafile.iloc[:, 3:], axis=1, inplace=False)
            # Rename columns
            data3D.rename(columns={0: 1, 1: 2, 2: 3}, inplace=True)
            # Find data dimensions
            datadimensions = data3D.shape
            datasize = datadimensions[0]
            # Total deviation column data
            STDcol = datafile[9]
            # Rotate points for Brazil data
            data3D = data3D.reindex(columns=[1, 3, 2])
            data3D = data3D.rename(columns={1: 1, 3: 2, 2: 3}, inplace=False)
            data3D[1] = -1*data3D[1]

            # ------------------Finding Height of Torso----------------------#
            maxy = max(data3D[2])
            miny = min(data3D[2])
            theight = maxy - miny
            # Plot with maxy and miny planes
            # fig = plt.figure()
            # ax = fig.add_subplot(4, 4, 1, projection='3d')
            # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.1)
            # axis1 = np.linspace(-300, 300, 300)
            # axis2 = np.linspace(-300, 300, 300)
            # plane1, plane2 = np.meshgrid(axis1, axis2)
            # ax.plot_surface(X=plane1, Y=float(maxy), Z=plane2, color='g', alpha=0.6)
            # ax.plot_surface(X=plane1, Y=float(miny), Z=plane2, color='b', alpha=0.6)
            
            # ------------------Finding Width of Torso----------------------#
            maxx = max(data3D[1])
            minx = min(data3D[1])
            twidth = maxx - minx
            # Plot with maxx and minx planes
            # ax = fig.add_subplot(4, 4, 2, projection='3d')
            # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.1)
            # ax.plot_surface(X=float(maxx), Y=plane1, Z=plane2, color='g', alpha=0.6)
            # ax.plot_surface(X=float(minx), Y=plane1, Z=plane2, color='b', alpha=0.6)
            
            # ------------------Finding Depth of Torso----------------------#
            maxz = max(data3D[3])
            minz = min(data3D[3])
            tdepth = maxz - minz
            # Plot with maxz, miny planes
            #
            #
            # Plot with maxx, maxy, maxz, minx, miny, minz planes
            # ax = fig.add_subplot(4, 4, 4, projection='3d')
            # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.1)
            # ax.plot_surface(X=plane1, Y=float(maxy), Z=plane2, color='g', alpha=0.2)
            # ax.plot_surface(X=plane1, Y=float(miny), Z=plane2, color='b', alpha=0.2)
            # ax.plot_surface(X=float(maxx), Y=plane1, Z=plane2, color='r', alpha=0.2)
            # ax.plot_surface(X=float(minx), Y=plane1, Z=plane2, color='y', alpha=0.2)

            # ------------------Adjust Torso Alignment----------------------#

            # ---------------Finding Back Point of the Torso-----------------#
            # Finding mean x value
            centerx = sum(data3D[1]) / len(data3D[1])
            # Finding mean y value
            centery = sum(data3D[2]) / len(data3D[2])
            # Finding mean z value
            centerz = sum(data3D[3]) / len(data3D[3])
            # Plot
            # ax = fig.add_subplot(4, 4, 5, projection='3d')
            # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.05)
            # ax.scatter(centerx, centery, centerz, color='r', alpha=1)
           
            yzclosestpoint = data3D
            # Finds all points that x-values are 4 apart from mean
            yzclosestpoint = yzclosestpoint[yzclosestpoint[1] < (centerx + 4)]
            yzclosestpoint = yzclosestpoint[yzclosestpoint[1] > (centerx - 4)]
            # Finds points with z values greater than mean so all back points are eliminated
            B6 = yzclosestpoint
            B6 = B6[B6[3] > centerz]
            # Plot
            # ax = fig.add_subplot(4, 4, 6, projection='3d')
            # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.05)
            # ax.scatter(B6[1], B6[2], B6[3], color='r', alpha=1)
            
            # Finds the lowest point on the back
            B4 = B6[2].idxmin()
            B4 = B6.loc[[B4]]
            # Plot with lowest back point
            # ax = fig.add_subplot(4, 4, 7, projection='3d')
            # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.05)
            # ax.scatter(B4[1], B4[2], B4[3], color='r', alpha=1)
            # Plot with centerx plane, yzclosest points, B4
            #

            # ---------------Move Origin to Back Point-----------------#
            x_ones = np.ones((datasize, 1))
            data3D_ones = np.hstack((data3D, x_ones))
            Translate_matrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [float(-B4[1]), float(-B4[2]), float(-B4[3]), 1]])
            tpts = np.matmul(data3D_ones, Translate_matrix)[:, :-1]
            
            # ---------------Add STD Column to data3D-----------------#
            data3Dnewco = pd.DataFrame(tpts)
            data3Dnewco['STD'] = STDcol
            
            # ------------------Finding Height of Torso----------------------#
            B7 = max(tpts[2])
            B8 = min(tpts[2])
            dis1 = B7 - B8
            
            # ------------------Finding Width of Torso----------------------#
            B9 = max(tpts[1])
            B10 = min(tpts[1])
            dis3 = B9 - B10
            
            # ------------------Finding Depth of Torso----------------------#
            B11 = max(tpts[3])
            B12 = min(tpts[3])
            dis5 = B11 - B12

            # -------Separate positive and negative patches on left and right side--------#
            for counter1 in range(numberofthre):
                dataDCMRp = pd.DataFrame()
                dataDCMRn = pd.DataFrame()
                dataDCMLp = pd.DataFrame()
                dataDCMLn = pd.DataFrame()
                data3Dnewco1 = data3Dnewco.sort_values(by=1)
                for i in range(datasize):
                    if data3Dnewco1.iloc[i, 0] > 0:
                        if data3Dnewco1.iloc[i, 3] > Threshold1[counter1]:
                             dataDCMRp = dataDCMRp.append(data3Dnewco1.iloc[i])
                        elif data3Dnewco1.iloc[i, 3] < -Threshold1[counter1]:
                            dataDCMRn = dataDCMRn.append(data3Dnewco1.iloc[i])
                    elif data3Dnewco1.iloc[i, 0] < 0:
                        if data3Dnewco1.iloc[i, 3] > Threshold1[counter1]:
                             dataDCMLp = dataDCMLp.append(data3Dnewco1.iloc[i])
                        elif data3Dnewco1.iloc[i, 3] < -Threshold1[counter1]:
                             dataDCMLn = dataDCMLn.append(data3Dnewco1.iloc[i])

            #WORK IN PROGRESS
            # # -------Separate positive and negative patches on left and right side--------#
            # ss = {'Rp0': "Rp", 'Rn0': "Rn", 'Lp0': "Lp", 'Ln0': "Ln"}
            # dataDCMall = {}
            # for counter1 in range(numberofthre):
            #     for i, j in ss.items():
            #         dataDCMall["dataDCM"+j]=pd.DataFrame()
            #     # for k,l in dataDCMall.items():
            #     #     exec(f"{k}=l")
            #
            #     data3Dnewco1 = data3Dnewco.sort_values(by=1)
            #     for k in range(datasize):
            #         if data3Dnewco1.iloc[k, 0] > 0:
            #             if data3Dnewco1.iloc[k, 3] > Threshold1[counter1]:
            #                  dataDCMall["dataDCMRp"] = dataDCMall["dataDCMRp"].append(data3Dnewco1.iloc[k])
            #             elif data3Dnewco1.iloc[k, 3] < -Threshold1[counter1]:
            #                 dataDCMall["dataDCMRn"] = dataDCMall["dataDCMRn"].append(data3Dnewco1.iloc[k])
            #         elif data3Dnewco1.iloc[k, 0] < 0:
            #             if data3Dnewco1.iloc[k, 3] > Threshold1[counter1]:
            #                  dataDCMall["dataDCMLp"] = dataDCMall["dataDCMLp"].append(data3Dnewco1.iloc[k])
            #             elif data3Dnewco1.iloc[k, 3] < -Threshold1[counter1]:
            #                  dataDCMall["dataDCMLn"] = dataDCMall["dataDCMLp"].append(data3Dnewco1.iloc[k])
            #     print(dataDCMall)
            #
            #
            #     # -------Separate individual patches--------# #WIP will change
            #     # for i, j in dataDCMall.items():
            #     red = [1.0, 0.0, 0.0]
            #     gray = [0.5, 0.5, 0.5]
            #
            #     arrayDCMRp = dataDCMall["dataDCMRp"].iloc[:, 0:3].to_numpy()
            #     cloudDCMRp = o3.geometry.PointCloud()
            #     cloudDCMRp.points = o3.utility.Vector3dVector(arrayDCMRp)
            #     cloudDCMRp.estimate_normals()
            #     cloudDCMRp.orient_normals_consistent_tangent_plane(4)
            #     cloudDCMRp.paint_uniform_color(red)
            #     #depth = 5
            #     #distance = np.asarray(cloudDCMRp.compute_nearest_neighbor_distance())
            #     mu_distance = 4 #np.mean(distance)
            #     radii_list = o3.utility.DoubleVector(np.array([0.50, 1.00, 1.5, 2.00]) * mu_distance)
            #
            #     meshDCMRp = o3.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=cloudDCMRp,
            #                                                                           radii=radii_list)
            #     meshDCMRp.compute_triangle_normals()
            #     meshDCMRp.compute_vertex_normals()
            #
            #     meshDCMRp.paint_uniform_color(gray)
            #     vis = o3.visualization.Visualizer()
            #     vis.create_window()
            #     vis.add_geometry(meshDCMRp)
            #     vis.add_geometry(cloudDCMRp)
            #     ren = vis.get_render_option()
            #     ren.mesh_show_wireframe = True
            #     ren.mesh_show_back_face = True
            #     ren.point_size = 5
            #     vis.run()
            #     #print(meshDCMRp.get_surface_area())
            #
            #     numberofvertices = np.asarray(meshDCMRp.vertices).shape[0]
            #
            #
            #     while numberofvertices > 0:
            #
            #         triangle_clusters, cluster_n_triangles, cluster_area = (
            #             meshDCMRp.cluster_connected_triangles())
            #         triangle_clusters = np.asarray(triangle_clusters)
            #         cluster_n_triangles = np.asarray(cluster_n_triangles)
            #         cluster_area = np.asarray(cluster_area)
            #
            #         print(triangle_clusters)
            #         print(cluster_n_triangles)
            #         print(cluster_area)
            #
            #         triangles_to_remove = cluster_n_triangles[triangle_clusters] < 20
            #         meshDCMRp.remove_triangles_by_mask(triangles_to_remove)
            #         o3.visualization.draw_geometries([meshDCMRp])
            #
            #         triangle_clusters, cluster_n_triangles, cluster_area = (
            #             meshDCMRp.cluster_connected_triangles())
            #         triangle_clusters = np.asarray(triangle_clusters)
            #         cluster_n_triangles = np.asarray(cluster_n_triangles)
            #         cluster_area = np.asarray(cluster_area)
            #
            #
            #         mesh_0=copy.deepcopy(meshDCMRp)
            #         lstccmpRp = []
            #         remove_all = []
            #         for i in range(0, len(cluster_n_triangles)):
            #             meshtemp = copy.deepcopy(mesh_0)
            #             remove_rest = triangle_clusters != i
            #             remove_all.append(triangle_clusters == i)
            #             meshtemp.remove_triangles_by_mask(remove_rest)
            #             meshtemp.remove_unreferenced_vertices()
            #             arraytemp=np.asarray(meshtemp.vertices).tolist()
            #             lstccmpRp.append(arraytemp)
            #         remove_any=np.any(remove_all, axis=0)
            #         meshDCMRp.remove_triangles_by_mask(remove_any)
            #         meshDCMRp.remove_unreferenced_vertices()
            #
            #         numberofvertices = np.asarray(meshDCMRp.vertices).shape[0]
            #         print(lstccmpRp)
            #         print(numberofvertices)



            # Plot Output
            #plt.show()            
