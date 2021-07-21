# -------------------------Import Libraries------------------------------#
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
import os
import scipy.io
from scipy.optimize import minimize
from sympy import Point3D, Plane
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3
import copy
import math

# -------------------------Set initial values------------------------------#
Threshold1 = [3, 9.33]
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
            data3D0 = datafile.drop(datafile.iloc[:, 3:], axis=1, inplace=False)
            # Rename columns
            data3D0.rename(columns={0: 1, 1: 2, 2: 3}, inplace=True)
            # Find data dimensions
            datadimensions = data3D0.shape
            datasize = datadimensions[0]
            # Total deviation column data
            STDcol = datafile[9]

            # ---------------Adjust alignment (aligns best plane of symmetry with the yz-plane)--------------------#

            bfmatfile = open(subdir + os.sep + 'bfmat.tfm', 'r').read().split()[:12] # read best fit alignment transformation matrix
            bfmatfile = [float(i) for i in bfmatfile]
            bfmat = np.reshape(np.array(bfmatfile), [3, 4])
            Q = bfmat[:, :3]
            c = bfmat[:, 3]
            # Fixed point on plane
            x = np.matmul(np.linalg.inv(np.identity(3) - Q), c)

            # define square error function
            def optimize_transform(r):
                p1 = r[0]
                p2 = r[1]
                p3 = r[2]
                th = r[3]
                Qq = np.array([[-p1 ** 2 * (1 + np.cos(th)) + np.cos(th), -p1 * p2 * (1 + np.cos(th)) + p3 * np.sin(th),
                                -p1 * p3 * (1 + np.cos(th)) - p2 * np.sin(th)],
                               [-p1 * p2 * (1 + np.cos(th)) - p3 * np.sin(th), -p2 ** 2 * (1 + np.cos(th)) + np.cos(th),
                                -p2 * p3 * (1 + np.cos(th)) + p1 * np.sin(th)],
                               [-p1 * p3 * (1 + np.cos(th)) + p2 * np.sin(th),
                                -p2 * p3 * (1 + np.cos(th)) - p1 * np.sin(th),
                                -p3 ** 2 * (1 + np.cos(th)) + np.cos(th)]])
                square_error = 0
                for m, n in np.nditer([Qq, Q]):
                    square_error += (m - n) ** 2
                return square_error

            # function to create a plane
            def create_plane(normalvector, twidth, theight, x, y):
                patchlimitsy = 0.67
                patchlimitsx = 0.35
                return twidth * normalvector[0] * (x - patchlimitsx) + theight * normalvector[1] * (y - patchlimitsy)
            
            # Best plane of symmetry
            result_opt = minimize(optimize_transform, [-1, 0, 0, 0])
            # Normal vector to plane
            p = result_opt.x[0:3]
            if p[0] > 0:
                p = -p

            # Transformation matrix
            u = np.array([-1, 0, 0])
            v = np.cross(p, u)
            T_cos = np.dot(p, u)
            v_skew = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
            R = np.identity(3) + v_skew + np.linalg.matrix_power(v_skew, 2) / (1 + T_cos)  # rotation matrix
            B = [0, 0, (p[0] * x[0] + p[1] * x[1]) / p[2] + x[2]]  # vertical axis intercept
            t = B - np.matmul(R, B)  # translation vector
            R_t = np.append(R, t.reshape(-1, 1), axis=1)
            T = np.zeros((4, 4))
            T[3, 3] = 1
            T[:-1, :] = R_t

            # Apply transformation
            dat = data3D0.transpose()
            dat = np.append(dat, np.ones((1, dat.shape[1])), axis=0)
            dat = np.matmul(T, dat)
            data3D = dat[0:3].transpose()

            # ------------------Rotate points for brazil data----------------------#
            data3D = pd.DataFrame(data3D, columns=[1, 2, 3])
            data3D = data3D.reindex(columns=[1, 3, 2])
            data3D.rename(columns={1: 1, 3: 2, 2: 3}, inplace=True)
            data3D[1] = -1 * data3D[1]

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
                                         [0, float(-B4[2]), float(-B4[3]), 1]])
            tpts = np.matmul(data3D_ones, Translate_matrix)[:, :-1]

            # ---------------Add STD Column to data3D-----------------#
            data3Dnewco = pd.DataFrame(tpts)
            data3Dnewco['STD'] = STDcol

            # ------------------Finding Height of Torso----------------------#
            maxy = max(tpts[2])
            miny = min(tpts[2])
            theight = maxy - miny
            # Plot with maxy and miny planes
            # fig = plt.figure()
            # ax = fig.add_subplot(4, 4, 1, projection='3d')
            # ax.scatter(tpts[1], tpts[2], tpts[3], c='tab:gray', alpha=0.1)
            # axis1 = np.linspace(-300, 300, 300)
            # axis2 = np.linspace(-300, 300, 300)
            # plane1, plane2 = np.meshgrid(axis1, axis2)
            # ax.plot_surface(X=plane1, Y=float(maxy), Z=plane2, color='g', alpha=0.6)
            # ax.plot_surface(X=plane1, Y=float(miny), Z=plane2, color='b', alpha=0.6)

            # ------------------Finding Width of Torso----------------------#
            maxx = max(tpts[1])
            minx = min(tpts[1])
            twidth = maxx - minx
            # Plot with maxx and minx planes
            # ax = fig.add_subplot(4, 4, 2, projection='3d')
            # ax.scatter(tpts[1], tpts[2], tpts[3], c='tab:gray', alpha=0.1)
            # ax.plot_surface(X=float(maxx), Y=plane1, Z=plane2, color='g', alpha=0.6)
            # ax.plot_surface(X=float(minx), Y=plane1, Z=plane2, color='b', alpha=0.6)

            # ------------------Finding Depth of Torso----------------------#
            maxz = max(tpts[3])
            minz = min(tpts[3])
            tdepth = maxz - minz
            # Plot with maxz, miny planes
            #
            #
            # Plot with maxx, maxy, maxz, minx, miny, minz planes
            # ax = fig.add_subplot(4, 4, 4, projection='3d')
            # ax.scatter(tpts[1], tpts[2], tpts[3], c='tab:gray', alpha=0.1)
            # ax.plot_surface(X=plane1, Y=float(maxy), Z=plane2, color='g', alpha=0.2)
            # ax.plot_surface(X=plane1, Y=float(miny), Z=plane2, color='b', alpha=0.2)
            # ax.plot_surface(X=float(maxx), Y=plane1, Z=plane2, color='r', alpha=0.2)
            # ax.plot_surface(X=float(minx), Y=plane1, Z=plane2, color='y', alpha=0.2)

            # -------Separate positive and negative patches on left and right side--------#

            for threshold in Threshold1:
                dataDCM = {"Rp": pd.DataFrame(), "Rn": pd.DataFrame(), "Lp": pd.DataFrame(), "Ln": pd.DataFrame()}
                # for k,l in dataDCM.items():
                #     exec(f"{k}=l")

                data3Dnewco1 = data3Dnewco.sort_values(by=1)
                for k in range(datasize):
                    if data3Dnewco1.iloc[k, 0] > 0:
                        if data3Dnewco1.iloc[k, 3] > threshold:
                            dataDCM["Rp"] = dataDCM["Rp"].append(data3Dnewco1.iloc[k])
                        elif data3Dnewco1.iloc[k, 3] < -threshold:
                            dataDCM["Rn"] = dataDCM["Rn"].append(data3Dnewco1.iloc[k])
                    elif data3Dnewco1.iloc[k, 0] < 0:
                        if data3Dnewco1.iloc[k, 3] > threshold:
                            dataDCM["Lp"] = dataDCM["Lp"].append(data3Dnewco1.iloc[k])
                        elif data3Dnewco1.iloc[k, 3] < -threshold:
                            dataDCM["Ln"] = dataDCM["Ln"].append(data3Dnewco1.iloc[k])
                # print(dataDCM)

                dictDCM = {"Rp": {}, "Rn": {}, "Lp": {}, "Ln": {}}
                centroid = {"Rp": [], "Rn": [], "Lp": [], "Ln": []} # for trcentrRp, trcentrLp, ...
                ccmp = {"Rp": [], "Rn": [], "Lp": [], "Ln": []} # for ccmpRp, ccmpLp, ...
                area = {"Rp": [], "Rn": [], "Lp": [], "Ln": []} # for AreaofeachTorsoRp, AreaofTorsoLp, ... (before cropping)
                normalx = {"Rp": [], "Rn": [], "Lp": [], "Ln": []} # for NormalxRp, NormalxLp, ...
                normaly = {"Rp": [], "Rn": [], "Lp": [], "Ln": []} # for NormalyRp, NormalyLp, ...
                normalz = {"Rp": [], "Rn": [], "Lp": [], "Ln": []} # for NormalzRp, NormalzLp, ...

                for i, DCM in dataDCM.items():
                    # Create dictionary to look up deviations
                    tuplesDCM = list(DCM[[0, 1, 2]].itertuples(index=False, name=None))
                    # dictDCM[i] = DCM.set_index([0, 1, 2]).T.to_dict('records')[0]
                    dictDCM[i] = {tup: list(DCM['STD'])[k] for k, tup in enumerate(tuplesDCM)}

                    # -------Build patch meshes--------#
                    red = [1.0, 0.0, 0.0]
                    gray = [0.5, 0.5, 0.5]

                    arrayDCM = DCM.iloc[:, 0:3].to_numpy()
                    cloudDCM = o3.geometry.PointCloud()
                    cloudDCM.points = o3.utility.Vector3dVector(arrayDCM)
                    cloudDCM.estimate_normals()
                    cloudDCM.orient_normals_consistent_tangent_plane(4)
                    cloudDCM.paint_uniform_color(red)
                    # depth = 5
                    # distance = np.asarray(cloudDCMRp.compute_nearest_neighbor_distance())
                    mu_distance = 4  # np.mean(distance)
                    radii_list = o3.utility.DoubleVector(np.array([0.50, 1.00, 1.5, 2.00]) * mu_distance)
                    meshDCM = o3.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=cloudDCM,
                                                                                             radii=radii_list)
                    meshDCM.compute_triangle_normals()
                    meshDCM.compute_vertex_normals()
                    meshDCM.paint_uniform_color(gray)
                    o3.visualization.draw_geometries([meshDCM, cloudDCM], mesh_show_wireframe=True,
                                                     mesh_show_back_face=True)
                    # print(meshDCM.get_surface_area())

                    triangle_clusters0, cluster_n_triangles0, cluster_area0 = (
                        meshDCM.cluster_connected_triangles())
                    triangle_clusters0 = np.asarray(triangle_clusters0)
                    cluster_n_triangles0 = np.asarray(cluster_n_triangles0)
                    cluster_area0 = np.asarray(cluster_area0)

                    # Filtering small patches
                    triangles_small_n = cluster_n_triangles0[triangle_clusters0] < 10
                    meshDCM.remove_triangles_by_mask(triangles_small_n)
                    o3.visualization.draw_geometries([meshDCM], mesh_show_wireframe=True, mesh_show_back_face=True)

                    triangle_clusters, cluster_n_triangles, cluster_area = (
                        meshDCM.cluster_connected_triangles())
                    triangle_clusters = np.asarray(triangle_clusters)
                    cluster_n_triangles = np.asarray(cluster_n_triangles)
                    cluster_area = np.asarray(cluster_area)

                    # -------Separate individual patches--------#
                    mesh_all = copy.deepcopy(meshDCM)
                    remove_idx = []
                    for k, surface_area in enumerate(cluster_area):
                        mesh_single = copy.deepcopy(mesh_all)
                        remove_rest = triangle_clusters != k
                        remove_idx.append(triangle_clusters == k)
                        mesh_single.remove_triangles_by_mask(remove_rest)
                        mesh_single.remove_unreferenced_vertices()
                        # o3.visualization.draw_geometries([mesh_single], mesh_show_wireframe=True,
                        #                                  mesh_show_back_face=True)
                        array_single = np.asarray(mesh_single.vertices).tolist()
                        # patch centroids
                        centroid[i].append(np.mean(array_single, axis=0))
                        for point in array_single:
                            point.append(dictDCM[i][tuple(point)])
                        # patch points + deviations
                        ccmp[i].append(array_single)
                        # patch surface area
                        area[i].append(surface_area)
                    remove_all = np.any(remove_idx, axis=0)
                    meshDCM.remove_triangles_by_mask(remove_all)
                    meshDCM.remove_unreferenced_vertices()

                    # Normalize location of patch centroids
                    normalx[i] = (np.asarray(centroid[i])[:, 0] / twidth).tolist()
                    normaly[i] = (np.asarray(centroid[i])[:, 1] / theight).tolist()
                    normalz[i] = (np.asarray(centroid[i])[:, 2] / tdepth).tolist()

                    # Plot Output
                    # plt.show()

                    # -------Filter false patches at waist, neck, and shoulders--------#

                            patchlimitLy = 0.05
                            patchlimitUy = 0.88

                            patchlimitSy = 0.67
                            patchlimitSx = 0.35

                            Splaneangle = 20

                            # R
                            SnormR = [math.cos(math.radians(Splaneangle)), math.sin(math.radians(Splaneangle))]
                            SvectR = [math.cos(math.radians(90 + Splaneangle)), math.sin(math.radians(90 + Splaneangle))]

                            # L
                            SnormL = [(-1) * math.cos(math.radians(Splaneangle)), math.sin(math.radians(Splaneangle))]
                            SvectL = [(-1) * math.cos(math.radians(90 + Splaneangle)), math.sin(math.radians(90 + Splaneangle))]

                            ccmp2 = {"Rp": [], "Rn": [], "Lp": [], "Ln": []}
                            types = ["Rp", "Rn", "Lp", "Ln"]

                            for i in range(len(types)):
                                for j in range(len(normaly[types[i]])):
                                    if normaly[types[i]][j] > patchlimitLy and (normaly[types[i]][j] < patchlimitSy or (
                                            normaly[types[i]][j] < patchlimitUy and create_plane(SnormR, twidth, theight,
                                                                                                 normalx[types[i]][j],
                                                                                                 normaly[types[i]][j]) < 0)):
                                        ccmp2[types[i]].append(ccmp[types[i]][j])
                                
            
            # ---------------------Transferring to Excel----------------------------#
            # IW: Not sure if all these variables are necessary, you can remove any that aren't actually needed
            # IW: Again, keep in mind how the data is stored differently than in Mathematica, see lines 216-221 for example
            # Creating names for the indices
            Firstrow = ["RMS+", "MaxDev+", "Area+", "Normalx+", "Normaly+", "Normalz+",
                        "Location+", "RMS-", "MaxDev-", "Area-", "Normalx-", "Normaly-",
                         "Normalz-", "Location-"]
            # Finding the max length for the rows      
            Rownu1 = max([len(RMSRp), len(MaxDevRp), len(AreaofeachTorsoRp), len(NormalxRp2),
                          len(NormalyRp2), len(NormalzRp2), len(TorsoRp), len(RMSRn),
                          len(MaxDevRn), len(AreaofeachTorsoRn), len(NormalxRn2), len(NormalyRn2),
                          len(NormalzRn2), len(TorsoRn)])
            # Finding max columns
            Colnu = len(Firstrow)
            # Create array
            resultsR = [RMSRp, MaxDevRp, AreaofeachTorsoRp, NormalxRp2, NormalyRp2, NormalzRp2, TorsoRp, RMSRn,
                        MaxDevRn, AreaofeachTorsoRn, NormalxRn2, NormalyRn2, NormalzRn2, TorsoRn]
            Rownu2 = max([len(RMSLp), len(MaxDevLp), len(AreaofeachTorsoLp), len(NormalxLp2),
                          len(NormalyLp2), len(NormalzLp2), len(TorsoLp), len(RMSLn),
                          len(MaxDevLn), len(AreaofeachTorsoLn), len(NormalxLn2), len(NormalyLn2),
                          len(NormalzLn2), len(TorsoLn)])
            resultsL = [RMSLp, MaxDevLp, AreaofeachTorsoLp, NormalxLp2, NormalyLp2, NormalzLp2, TorsoLp, RMSLn,
                        MaxDevLn, AreaofeachTorsoLn, NormalxLn2, NormalyLn2, NormalzLn2, TorsoLn]
            # Create DataFrame and Transpose
            tableR = pd.DataFrame(resultsR, index=Firstrow)
            # replace empty strings with 'P' /// not done
            tableR = tableR.T
            tableL = pd.DataFrame(resultsL, index=Firstrow)
            # replace empty strings with 'P' /// not done
            tableL = tableL.T
            # Find the folder that program is in for naming .csv files
            foldernum = subdir.split('\\')
            # create .csv files
#             tableR.to_csv(path + '\Features-Right-' + foldernum[1] +'3mm.csv', index=False)
#             tableR.to_csv(path + '\Features-Right-' + foldernum[1] + '9.33mm.csv', index=False)
#             tableL.to_csv(path + '\Features-Left-' + foldernum[1] +'3mm.csv', index=False)
#             tableL.to_csv(path + '\Features-Left-' + foldernum[1] + '9.33mm.csv', index=False)
