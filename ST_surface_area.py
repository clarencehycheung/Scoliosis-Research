import numpy as np
import open3d as o3
from tkinter.filedialog import askdirectory
import os
import pandas as pd
import sys
from io import StringIO

red = [1.0, 0.0, 0.0]
gray = [0.5, 0.5, 0.5]

thresholds = [3, 9.33]

features = {"Right": {"positive": pd.DataFrame(), "negative": pd.DataFrame()},
            "Left": {"positive": pd.DataFrame(), "negative": pd.DataFrame()}}
patch_sets = {"Right": {"positive": pd.DataFrame(), "negative": pd.DataFrame()},
              "Left": {"positive": pd.DataFrame(), "negative": pd.DataFrame()}}

# Prompting user to select the parent folder
path = askdirectory(title='Select Folder')

# Loops through the subfolders in the parent folder
for subdir, dirs, files in os.walk(path):
    for sub_folder in dirs:
        print('\nfolder: ' + sub_folder)
        filepath = subdir + os.sep + sub_folder + os.sep
        for threshold in thresholds:
            for side, signs in patch_sets.items():
                # read features .csv file and split into positive and negative datasets
                features_csv = pd.read_csv(
                    filepath + 'Features-' + side + '-' + sub_folder + '-' + str(threshold) + "mm.csv", header=0)
                curve_class = features_csv.loc[:, 'Curve Class (tree)']
                features_csv = features_csv.drop('Curve Class (tree)', axis=1)
                features[side]['positive'] = features_csv.drop(features_csv.iloc[:, 7:], axis=1)
                features[side]['negative'] = features_csv.drop(features_csv.iloc[:, :7], axis=1)

                for sign, data_set in signs.items():
                    if sign == 'positive':
                        sign_sym = '+'
                    else:
                        sign_sym = '-'
                    features[side][sign].rename({'Area' + sign_sym: 'No. points' + sign_sym}, axis=1, inplace=True)
                    features[side][sign] = features[side][sign][
                        features[side][sign]['RMS' + sign_sym] != 'P']  # delete empty 'P' rows
                    features[side][sign].iloc[:, 0:6] = features[side][sign].iloc[:, 0:6].astype(float)

                    # read patches .dat file as DataFrame
                    with open(filepath + side + ' ' + sign + ' patches-' + sub_folder + '-' + str(threshold) + 'mm.dat',
                              'r') as file_dat:
                        patches_dat = file_dat.readlines()
                        area_list = []
                        for i, patch in enumerate(patches_dat):
                            digit_idx = [(j, char) for j, char in enumerate(patch) if char.isdigit()]
                            patch_str = patch[digit_idx[0][0]:digit_idx[-1][0] + 1]
                            patch_str = patch_str.replace('*^', 'e')
                            patch_sets[side][sign] = pd.DataFrame(
                                [row.split(',') for row in patch_str.split('}\t{')]).astype(float)

                            # create mesh using Open3D ball pivoting method
                            coordinates = patch_sets[side][sign].iloc[:, 0:3].to_numpy()
                            point_set = o3.geometry.PointCloud()
                            point_set.points = o3.utility.Vector3dVector(coordinates)
                            point_set.estimate_normals()
                            point_set.orient_normals_consistent_tangent_plane(4)
                            # point_set.paint_uniform_color(red)
                            mu_distance = 3.5
                            radii_list = o3.utility.DoubleVector(np.array([2.5, 3.0]) * mu_distance)
                            mesh = o3.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=point_set,
                                                                                                  radii=radii_list)
                            mesh.compute_triangle_normals()
                            mesh.compute_vertex_normals()
                            # mesh.paint_uniform_color(gray)
                            # o3.visualization.draw_geometries([mesh, point_set], mesh_show_wireframe=True,
                            #                                  mesh_show_back_face=True)
                            area_list.append(mesh.get_surface_area())
                    features[side][sign].insert(2, 'Area' + sign_sym, area_list)  # insert new area column

                # save results to .csv
                result = pd.concat([features[side]['positive'], features[side]['negative']], axis=1)
                result.fillna('P', inplace=True)
                result = pd.concat([result, curve_class], axis=1)
                result.to_csv(
                    filepath + 'Features-Py-area-' + side + '-' + sub_folder + '-' + str(threshold) + 'mm.csv',
                    index=False)
                print(side + '-' + sub_folder + '-' + str(threshold) + 'mm')
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                    print(result)
        print('next')
print('end')
