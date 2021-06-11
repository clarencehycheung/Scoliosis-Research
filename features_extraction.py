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


            # Finding height of torso
            maxy = data3D.iloc[data3D[2].idxmax()]
            maxy = [maxy.iloc[0], maxy.iloc[1], maxy.iloc[2]]
            miny = data3D.iloc[data3D[2].idxmin()]
            miny = [miny.iloc[0], miny.iloc[1], miny.iloc[2]]
            
            
            xzplane01 = Plane(Point3D(0, maxy, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
            xzplane02 = Plane(Point3D(0, miny, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
            # Find distance between xzplane02 and maxy
            theight = maxy3D.distance(xzplane02)



            # Finding width of torso
            maxx = data3D.iloc[data3D[1].idxmax()]
            maxx = [maxx.iloc[0], maxx.iloc[1], maxx.iloc[2]]
            minx = data3D.iloc[data3D[1].idxmin()]
            minx = [minx.iloc[0], minx.iloc[1], minx.iloc[2]]
    
            
            yzplane01 = Plane(Point3D(0, maxx, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
            yzplane02 = Plane(Point3D(0, minx, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
            # Find distance between yzplane02 and maxx
            twidth = maxx3D.distance(yzplane02)

            # Finding depth of torso
            maxz = data3D.iloc[data3D[3].idxmax()]
            maxz = [maxz.iloc[0], maxz.iloc[1], maxz.iloc[2]]
            minz = data3D.iloc[data3D[3].idxmin()]
            minz = [minz.iloc[0], minz.iloc[1], minz.iloc[2]]
            
            
            xyplane01 = Plane(Point3D(0, maxz, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
            xyplane02 = Plane(Point3D(0, minz, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
            # Find distance between xyplane02 and maxz
            tdepth = maxz3D.distance(xyplane02)
