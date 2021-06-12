# -------------------------Import Libraries------------------------------#
import numpy as np
import pandas as pd
from tkinter.filedialog import askdirectory
import os
from sympy import Point3D, Plane

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
