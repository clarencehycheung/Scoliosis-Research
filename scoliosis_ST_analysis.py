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
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import open3d as o3
import copy
import math
from itertools import compress
from cycler import cycler
import trimesh


# -------------------------Define functions------------------------------#
# square error function for optimization
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


# plane function for patch filtering
def create_plane(norm_vector, width, height, x, y, x_limit, y_limit):
    return width * norm_vector[0] * (x - x_limit) + height * norm_vector[1] * (y - y_limit)


def RMS_fn(x):
    return np.sqrt(np.mean(np.asarray(x)[:, 3] ** 2))


def max_dev_fn(x, i):
    if i[1] == "p":
        def max_dev(y):
            return max(np.asarray(y)[:, 3])
    else:
        def max_dev(y):
            return min(np.asarray(y)[:, 3])
    return map(max_dev, x)


def locate(y):
    if y < 0.33:
        return "L"
    else:
        return "T-L"


def meshDevs(refMesh):
    """  Use .ply files
    refMesh is the reference mesh that is not being reflected and is fixed
    testMesh is the refMesh to be reflected and aligned to the refMesh
    testMeshPCDTransfrmd is the point cloud of the testMesh which is aligned on to the point cloud of the refMesh
    devs is a ndvector containing the signed deviations (magnitudes) from each point of the refMesh's point cloud to the closest point
    of testMeshPCDTransfrmd
    """

    refMesh_center = refMesh.get_center()

    T_reflect = np.eye(4)
    T_translt = np.eye(4)
    T_reflect[0, 0] = -1  # reflection with respect to the Y-Z plane

    testMesh = copy.deepcopy(refMesh).transform(T_reflect)
    testMesh_center = testMesh.get_center()
    T_translt[0:3, 3] = refMesh_center - testMesh_center
    testMesh.transform(T_translt)

    refPCD = o3.geometry.PointCloud()
    refPCD.points = refMesh.vertices
    movingPCD = o3.geometry.PointCloud()  # a point cloud that is going under translation and rotation to be aligned on the refPCD
    movingPCD.points = testMesh.vertices

    maxThrsh = np.linalg.norm(refPCD.get_max_bound() - refPCD.get_min_bound()) / 20
    maxIter = 2000
    threshold = maxThrsh  # max_correspondence_distance, i.e. the maximum distance in which the search tries to find a correspondence for each point

    reg_p2p = o3.pipelines.registration.registration_icp(movingPCD, refPCD, threshold, np.eye(4),
                                                         o3.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         o3.pipelines.registration.ICPConvergenceCriteria(
                                                             max_iteration=maxIter))
    testMeshPCDTransfrmd = movingPCD.transform(reg_p2p.transformation)
    # testMesh.transform(reg_p2p.transformation)

    # refPCD.paint_uniform_color([1, 0.706, 0]) # orange
    # testMeshPCDTransfrmd.paint_uniform_color([0, 0.651, 0.929])
    # refMesh.paint_uniform_color([1, 0.706, 0])  # orange
    # o3.visualization.draw_geometries([refPCD, testMeshPCDTransfrmd], height=600, width=860)
    # o3.visualization.draw_geometries([refMesh, testMeshPCDTransfrmd], height=600, width=860)

    # orienting reference mesh i.e. computing the reference mesh normals at vertices and flipping them if not pointing in an outward sense
    refVerts = np.asarray(refMesh.vertices)
    refMesh.compute_vertex_normals(normalized=True)
    refVertNormals = np.asarray(refMesh.vertex_normals)
    randVIndx = []
    tmp = np.random.randint(0, 100)
    for x in range(20):
        while tmp in randVIndx:
            tmp = np.random.randint(0, 100)
        randVIndx.append(tmp)
    randVIndx.sort()

    isOutward = sum([np.inner(refVerts[i] - refMesh_center, refVertNormals[i]) >= 0.0 for i in randVIndx]) > 15

    if (isOutward == False):
        refVertNormals = -1.0 * refVertNormals
        # refMesh.vertex_normals = o3.utility.Vector3dVector(refVertNormals)

    # Calculating the signed distance from each point of the refMesh to the test mesh
    targetPoints = np.asarray(testMeshPCDTransfrmd.points)
    numOfvert = refVerts.shape[0]
    devs = []
    pcd_tree = o3.geometry.KDTreeFlann(testMeshPCDTransfrmd)
    for i in range(numOfvert):
        refVert = refVerts[i]
        unitNrmVec = refVertNormals[i]
        k, cPInd, _ = pcd_tree.search_knn_vector_3d(refVert, 1)
        ind = cPInd[0]
        distVec = targetPoints[ind] - refVerts[i]
        normalDistVec = np.inner(distVec, unitNrmVec)
        devs.append(normalDistVec)

    transform = np.matmul(reg_p2p.transformation, T_translt)

    # if (np.inner(distVec , unitNrmVec) < 0):
    #    devs.append(-1 * np.linalg.norm(distVec))
    # else:
    #    devs.append(np.linalg.norm(distVec))

    return transform, np.array(devs)


# ------------------------------------------------------------------------------------------------------#
# Define deviation thresholds
thresholds = [3, 9.33]

# labels and colors for plotting
ccmp_label = {'Rp': 'Right Positive', 'Rn': 'Right Negative', 'Lp': 'Left Positive', 'Ln': 'Left Negative'}
center_color = {'Rp': '#FF39E0', 'Rn': '#FF39E0', 'Lp': '#00FFE4', 'Ln': '#00FFE4'}

# Prompt user to select folder
path = askdirectory(title='Select Folder')

# walk through path and find open mesh file
for subdir, dirs, files in os.walk(path):
    file_path = subdir
    if not os.path.exists(file_path + r"\cropped mesh.ply"):
        print('mesh file not found')
        quit()
    else:
        # Convert .ply mesh to .glb
        mesh_in = trimesh.load(file_path + r"\cropped mesh.ply")
        mesh_in.export(file_path + r"\cropped mesh.glb")
        mesh_scene = trimesh.load(file_path + r"\cropped mesh.glb")
        mesh_tri = list(mesh_scene.geometry.values())[0]
        mesh_crop = mesh_tri.as_open3d
        # plot of cropped torso mesh
        # o3.visualization.draw_geometries([mesh_crop], mesh_show_wireframe=True, mesh_show_back_face=True)
        data3D0 = pd.DataFrame(mesh_crop.vertices)
        # Rename columns
        data3D0.rename(columns={0: 1, 1: 2, 2: 3}, inplace=True)
        # Find data dimensions
        data_dimensions = data3D0.shape
        data_size = data_dimensions[0]
        # Return transformation matrix and total deviations
        bf_transform, STDcol = meshDevs(mesh_crop)

        # ---------------Export Deviation Map--------------------#
        # Define color map
        jet_map = cm.get_cmap('jet')
        dev_colors = [jet_map(0.05), jet_map(0.1), jet_map(0.2), jet_map(0.3), jet_map(0.4), (0, 1, 6 / 255, 1),
                      (0, 1, 6 / 255, 1), jet_map(0.6), jet_map(0.7), jet_map(0.8),
                      jet_map(0.9), jet_map(0.95)]
        color_scale = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9, 1]
        dev_map = LinearSegmentedColormap.from_list("deviation map", list(zip(color_scale, dev_colors)))

        # Apply color map and export as .glb
        mesh_tri.visual.vertex_colors = trimesh.visual.interpolate(STDcol, color_map=dev_map)
        # mesh_tri.show()
        mesh_tri.export(file_path + r"\deviation map.glb")

        # ---------------Adjust alignment (aligns best plane of symmetry with the yz-plane)--------------------#
        bf_mat = copy.deepcopy(bf_transform)  # best fit alignment transformation matrix
        bf_mat = bf_mat[:3, :]
        bf_mat[:, 0] = -bf_mat[:, 0]

        Q = bf_mat[:, :3]
        c = bf_mat[:, 3]
        # Fixed point on plane
        x = np.matmul(np.linalg.inv(np.identity(3) - Q), c)

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
        center_x = sum(data3D[1]) / len(data3D[1])
        # Finding mean y value
        center_y = sum(data3D[2]) / len(data3D[2])
        # Finding mean z value
        center_z = sum(data3D[3]) / len(data3D[3])
        # Plot
        # ax = fig.add_subplot(4, 4, 5, projection='3d')
        # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.05)
        # ax.scatter(center_x, center_y, center_z, color='r', alpha=1)

        yz_close_pts = data3D
        # Finds all points that x-values are 4 apart from 0
        yz_close_pts = yz_close_pts[abs(yz_close_pts[1]) < 4]
        # Finds points with z values greater than mean so all front points are eliminated
        yz_back_pts = yz_close_pts
        yz_back_pts = yz_back_pts[yz_back_pts[3] > center_z]
        # Plot
        # ax = fig.add_subplot(4, 4, 6, projection='3d')
        # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.05)
        # ax.scatter(yz_back_pts[1], yz_back_pts[2], yz_back_pts[3], color='r', alpha=1)

        # Finds the lowest point on the back
        min_pt_idx = yz_back_pts[2].idxmin()
        back_pt = yz_back_pts.loc[[min_pt_idx]]

        # Plot with lowest back point
        # ax = fig.add_subplot(4, 4, 7, projection='3d')
        # ax.scatter(data3D[1], data3D[2], data3D[3], c='tab:gray', alpha=0.05)
        # ax.scatter(back_pt[1], back_pt[2], back_pt[3], color='r', alpha=1)
        # Plot with center_x plane, yzclosest points, back_pt
        #

        # ---------------Move Origin to Back Point-----------------#
        x_ones = np.ones((data_size, 1))
        data3D_ones = np.hstack((data3D, x_ones))
        Translate_matrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, float(-back_pt[2]), float(-back_pt[3]), 1]])
        tpts = np.matmul(data3D_ones, Translate_matrix)[:, :-1]  # final coordinates

        # ---------------Add STD Column and tpts-----------------#
        tdata = pd.DataFrame(tpts, columns=["x", "y", "z"])
        tdata['STD'] = STDcol

        # ------------------Finding Height of Torso----------------------#
        max_y = max(tpts[:, 1])
        min_y = min(tpts[:, 1])
        t_height = max_y - min_y
        # Plot with max_y and min_y planes
        # fig = plt.figure()
        # ax = fig.add_subplot(4, 4, 1, projection='3d')
        # ax.scatter(tpts[1], tpts[2], tpts[3], c='tab:gray', alpha=0.1)
        # axis1 = np.linspace(-300, 300, 300)
        # axis2 = np.linspace(-300, 300, 300)
        # plane1, plane2 = np.meshgrid(axis1, axis2)
        # ax.plot_surface(X=plane1, Y=float(max_y), Z=plane2, color='g', alpha=0.6)
        # ax.plot_surface(X=plane1, Y=float(min_y), Z=plane2, color='b', alpha=0.6)

        # ------------------Finding Width of Torso----------------------#
        max_x = max(tpts[:, 0])
        min_x = min(tpts[:, 0])
        t_width = max_x - min_x
        # Plot with max_x and min_x planes
        # ax = fig.add_subplot(4, 4, 2, projection='3d')
        # ax.scatter(tpts[1], tpts[2], tpts[3], c='tab:gray', alpha=0.1)
        # ax.plot_surface(X=float(max_x), Y=plane1, Z=plane2, color='g', alpha=0.6)
        # ax.plot_surface(X=float(min_x), Y=plane1, Z=plane2, color='b', alpha=0.6)

        # ------------------Finding Depth of Torso----------------------#
        max_z = max(tpts[:, 2])
        min_z = min(tpts[:, 2])
        t_depth = max_z - min_z
        # Plot with max_z, min_y planes
        #
        #
        # Plot with max_x, max_y, max_z, min_x, min_y, min_z planes
        # ax = fig.add_subplot(4, 4, 4, projection='3d')
        # ax.scatter(tpts[1], tpts[2], tpts[3], c='tab:gray', alpha=0.1)
        # ax.plot_surface(X=plane1, Y=float(max_y), Z=plane2, color='g', alpha=0.2)
        # ax.plot_surface(X=plane1, Y=float(min_y), Z=plane2, color='b', alpha=0.2)
        # ax.plot_surface(X=float(max_x), Y=plane1, Z=plane2, color='r', alpha=0.2)
        # ax.plot_surface(X=float(min_x), Y=plane1, Z=plane2, color='y', alpha=0.2)

        # -------Separate positive and negative patches on left and right side--------#
        # loop through thresholds
        for th_idx, threshold in enumerate(thresholds):
            dataDCM = {"Rp": pd.DataFrame(), "Rn": pd.DataFrame(), "Lp": pd.DataFrame(), "Ln": pd.DataFrame()}
            # for k,l in dataDCM.items():
            #     exec(f"{k}=l")

            tdata_sorted = tdata.sort_values(by="y")

            condition_Rp = (tdata_sorted.x > 0) & (tdata_sorted.STD > threshold)
            condition_Rn = (tdata_sorted.x > 0) & (tdata_sorted.STD < -threshold)
            condition_Lp = (tdata_sorted.x < 0) & (tdata_sorted.STD > threshold)
            condition_Ln = (tdata_sorted.x < 0) & (tdata_sorted.STD < -threshold)

            dataDCM["Rp"] = tdata_sorted[condition_Rp]
            dataDCM["Rn"] = tdata_sorted[condition_Rn]
            dataDCM["Lp"] = tdata_sorted[condition_Lp]
            dataDCM["Ln"] = tdata_sorted[condition_Ln]
            # print(dataDCM)

            dictDCM = {"Rp": {}, "Rn": {}, "Lp": {}, "Ln": {}}
            centroid = {"Rp": [], "Rn": [], "Lp": [], "Ln": []}
            ccmp = {"Rp": [], "Rn": [], "Lp": [], "Ln": []}
            centroid2 = {"Rp": [], "Rn": [], "Lp": [], "Ln": []}  # for after patch filtering
            ccmp2 = {"Rp": [], "Rn": [], "Lp": [], "Ln": []}
            result = {"Rp": pd.DataFrame(), "Rn": pd.DataFrame(), "Lp": pd.DataFrame(), "Ln": pd.DataFrame()}
            result2 = {"Rp": pd.DataFrame(), "Rn": pd.DataFrame(), "Lp": pd.DataFrame(), "Ln": pd.DataFrame()}

            # loop through DCM datasets (Rp, Rn, Lp, Ln)
            for i, DCM in dataDCM.items():
                if i[1] == "p":
                    sign = '+'
                else:
                    sign = '-'

                # Create dictionary to look up deviations
                tuplesDCM = list(DCM[["x", "y", "z"]].itertuples(index=False, name=None))
                dictDCM[i] = {tup: list(DCM['STD'])[k] for k, tup in enumerate(tuplesDCM)}

                # -------Build patch meshes--------# can replace this section
                red = [1.0, 0.0, 0.0]
                gray = [0.5, 0.5, 0.5]

                arrayDCM = DCM.iloc[:, 0:3].to_numpy()
                cloudDCM = o3.geometry.PointCloud()
                cloudDCM.points = o3.utility.Vector3dVector(arrayDCM)
                cloudDCM.estimate_normals()
                cloudDCM.orient_normals_consistent_tangent_plane(4)
                cloudDCM.paint_uniform_color(red)
                mu_distance = 2
                radii_list = o3.utility.DoubleVector(np.array([4, 4.5]) * mu_distance)  # testing radius size
                meshDCM = o3.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=cloudDCM,
                                                                                         radii=radii_list)
                meshDCM.compute_triangle_normals()
                meshDCM.compute_vertex_normals()
                meshDCM.paint_uniform_color(gray)
                # plot of entire mesh
                # o3.visualization.draw_geometries([meshDCM, cloudDCM], mesh_show_wireframe=True,
                #                                  mesh_show_back_face=True)

                # print(meshDCM.get_surface_area()) # total surface area

                triangle_clusters0, cluster_n_triangles0, cluster_area0 = (
                    meshDCM.cluster_connected_triangles())
                triangle_clusters0 = np.asarray(triangle_clusters0)
                cluster_n_triangles0 = np.asarray(cluster_n_triangles0)
                cluster_area0 = np.asarray(cluster_area0)

                # Filtering small patches
                triangles_small_n = cluster_n_triangles0[triangle_clusters0] < 5  # testing minimum number of triangles
                meshDCM.remove_triangles_by_mask(triangles_small_n)
                # plot with small patches removed
                # o3.visualization.draw_geometries([meshDCM], mesh_show_wireframe=True, mesh_show_back_face=True)

                triangle_clusters, cluster_n_triangles, cluster_area = (
                    meshDCM.cluster_connected_triangles())
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                cluster_area = np.asarray(cluster_area)

                # -------Separate individual patches--------#
                mesh_all = copy.deepcopy(meshDCM)
                remove_idx = []
                area = []
                for k, surface_area in enumerate(cluster_area):
                    mesh_single = copy.deepcopy(mesh_all)
                    remove_rest = triangle_clusters != k
                    remove_idx.append(triangle_clusters == k)
                    mesh_single.remove_triangles_by_mask(remove_rest)
                    mesh_single.remove_unreferenced_vertices()
                    # plot of individual patches
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
                    area.append(surface_area)
                remove_all = np.any(remove_idx, axis=0)
                meshDCM.remove_triangles_by_mask(remove_all)
                meshDCM.remove_unreferenced_vertices()

                # plot them?

                # results before patch filtering
                # RMS deviation, Maximum deviation, patch area, and normalized centroid coordinates
                result[i] = pd.DataFrame({
                    "RMS" + sign: map(RMS_fn, ccmp[i]),
                    "Max Dev" + sign: max_dev_fn(ccmp[i], i),
                    "Area" + sign: area,
                    "Normal x" + sign: (np.asarray(centroid[i])[:, 0] / t_width),
                    "Normal y" + sign: (np.asarray(centroid[i])[:, 1] / t_height),
                    "Normal z" + sign: (np.asarray(centroid[i])[:, 2] / t_depth)
                })

                result[i]["Location" + sign] = list(map(locate, result[i]["Normal y" + sign]))

                # print(threshold, i)
                # print(result[i])

                # -------Filter false patches at waist, neck, and shoulders--------#
                # testing plane thresholds
                # upper and lower plane
                lower_limit_y = [0.04, 0.04]
                upper_limit_y = [0.92, 0.94]

                # shoulder plane
                shoulder_limit_y = [0.58, 0.58]
                shoulder_limit_x = [0.45, 0.46]
                shoulder_angle = 15

                if i[0] == "L":
                    shoulder_angle = 180 - shoulder_angle
                    shoulder_limit_x[th_idx] = -shoulder_limit_x[th_idx]

                shoulder_norm = [math.cos(math.radians(shoulder_angle)),
                                 math.sin(math.radians(shoulder_angle))]  # plane normal vector
                shoulder_tang = [math.cos(math.radians(90 + shoulder_angle)),
                                 math.sin(math.radians(90 + shoulder_angle))]  # plane tangent vector

                # results after patch filtering
                result2[i] = pd.DataFrame()

                condition_area = result[i]["Area" + sign] > 5000  # testing area threshold
                condition_l = result[i]["Normal y" + sign] > lower_limit_y[th_idx]
                condition_s = result[i]["Normal y" + sign] < shoulder_limit_y[th_idx]
                condition_u = result[i]["Normal y" + sign] < upper_limit_y[th_idx]
                condition_plane = create_plane(shoulder_norm, t_width, t_height,
                                               result[i]["Normal x" + sign],
                                               result[i]["Normal y" + sign],
                                               shoulder_limit_x[th_idx],
                                               shoulder_limit_y[th_idx]) < 0
                condition_all = (condition_area | (condition_l & (condition_s | (condition_u & condition_plane))))

                result2[i] = result[i][condition_all].reset_index(drop=True)
                centroid2[i] = list(compress(centroid[i], condition_all))
                ccmp2[i] = list(compress(ccmp[i], condition_all))

            # -------Final plot figures--------# WIP consider other method for plotting
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
            ax.set_box_aspect((np.ptp(-tpts[:, 0]), np.ptp(tpts[:, 2]), np.ptp(tpts[:, 1])))
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            ax.set_axis_off()

            # plot torso points
            ax.scatter(-tpts[:, 0], tpts[:, 2], tpts[:, 1], s=0.01, c='grey', alpha=0.9)

            # plot arrows as axis
            x0, y0, z0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            u0, v0, w0 = np.array(
                [[-max(abs(-tpts[:, 0])) - 30, 0, 0, max(abs(-tpts[:, 0])) + 30], [0, min(tpts[:, 2]) - 30, 0, 0],
                 [0, 0, max(tpts[:, 1]) + 30, 0]])
            ax.quiver(x0, y0, z0, u0, v0, w0, linewidth=0.5, arrow_length_ratio=0.025, color="black")

            # set patch colors
            patch_color_cycle = cycler(
                color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:olive',
                       '#A4ff00', '#800039', 'y'])
            ax.set_prop_cycle(patch_color_cycle)

            # plot patches and patch centroids
            for i, center in centroid2.items():
                if center:
                    center = np.asarray(center)
                    if i[1] == "p":
                        mark = "+"
                    else:
                        mark = "_"
                    ax.scatter(-center[:, 0], center[:, 2], center[:, 1], s=100, marker=mark, c=center_color[i],
                               linewidths=3, label=ccmp_label[i])
                    for j, patch in enumerate(ccmp2[i]):
                        patch = np.asarray(patch)
                        ax.scatter(-patch[:, 0], patch[:, 2], patch[:, 1], s=3, alpha=0.5, label='patch ' + str(j + 1))

            ax.legend(loc='right')
            plt.tight_layout()

            # back view
            ax.view_init(0, 90)
            # plt.show()
            plt.savefig(file_path + os.sep + str(threshold) + 'mm-torso-back-final.jpg')

            # side view
            ax.view_init(0, 180)
            # plt.show()
            plt.savefig(file_path + os.sep + str(threshold) + 'mm-torso-side-final.jpg')

            # perspective view
            ax.view_init(25, 120)
            ax.set_proj_type('persp')
            # plt.show()
            plt.savefig(file_path + os.sep + str(threshold) + 'mm-torso-persp-final.jpg')

            # create right and left result sets
            result_R = pd.concat([result2["Rp"], result2["Rn"]], axis=1)
            result_L = pd.concat([result2["Lp"], result2["Ln"]], axis=1)

            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                print("Right-" + str(threshold) + "mm")
                print(result_R)
                print("Left-" + str(threshold) + "mm")
                print(result_L)
                print()

            # save results to .csv
            result_R.to_csv(file_path + '\Features-Right-' + str(threshold) + 'mm.csv', index=False)
            result_L.to_csv(file_path + '\Features-Left-' + str(threshold) + 'mm.csv', index=False)
