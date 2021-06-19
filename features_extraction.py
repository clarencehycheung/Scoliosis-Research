# -------------------------Import Libraries------------------------------#
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
import os
from sympy import Point3D, Plane

# -------------------------Set initial values------------------------------#
Threshold1 = {3, 9.333}
initial = 1
initial1 = 1
numebrofthre = len(Threshold1)

# Prompting user to select the parent folder
path = askdirectory(title='Select Folder')

# Loops through the subfolders in the parent folder and finds the EDM-1.csv files
for subdir, dirs, files in os.walk(path):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith("EDM-1.csv"):
            datafile = pd.read_csv(filepath, header=None)
            # Delete columns
            datafile.drop([0, 1, 2, 6, 7, 8], axis=1, inplace=True)
            # Rename columns
            data3D = datafile.rename(columns={3: 1, 4: 2, 5: 3, 9: 4}, inplace=False)
            # Find data dimensions
            datadimensions = data3D.shape
            datasize = datadimensions[0]
            # Deviations column data
            STDcol = data3D[4]
            # Test coordinates
            data3D.drop([4], axis=1, inplace=True)
            # Rotate points for Brazil data
            data3D = data3D.reindex(columns=[1, 3, 2])
            data3D = data3D.rename(columns={1: 1, 3: 2, 2: 3}, inplace=False)
            data3D[1] = -1*data3D[1]

            # ------------------Finding Height of Torso----------------------#
            maxy = max(data3D[2])
            miny = min(data3D[2])
            theight = maxy - miny
            
            # ------------------Finding Width of Torso----------------------#
            maxx = max(data3D[1])
            minx = min(data3D[1])
            twidth = maxx - minx
            
            # ------------------Finding Depth of Torso----------------------#
            maxz = max(data3D[3])
            minz = min(data3D[3])
            tdepth = maxz - minz

            # ------------------Adjust Torso Alignment----------------------#

            # ---------------Finding Back Point of the Torso-----------------#
            # Finding mean x value
            centerx = sum(data3D[1]) / len(data3D[1])
            # Finding mean z value
            centerz = sum(data3D[3]) / len(data3D[3])
            yzclosestpoint = data3D
            # Finds all points that x-values are 4 apart from mean
            yzclosestpoint = yzclosestpoint[yzclosestpoint[1] < (centerx + 4)]
            yzclosestpoint = yzclosestpoint[yzclosestpoint[1] > (centerx - 4)]
            # Finds points with z values greater than mean so all back points are eliminated
            B6 = yzclosestpoint
            B6 = B6[B6[3] > centerz]
            # Finds the lowest point on the back
            B4 = B6[2].idxmin()
            B4 = B6.loc[[B4]]

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
            
            # -------Separate Positive and Negative Patches in New Coordinate--------#
            for counter1 in range(initial1, numebrofthre + initial1):
                dataDCMp = []
                dataDCMn = []
                data3Dnewco1 = data3Dnewco.sort_values(by=1)
                for i in range(datasize):
                    if data3Dnewco1.loc[i, 1] > 0 and ...............
