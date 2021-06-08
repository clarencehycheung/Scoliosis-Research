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
            data3D = datafile.rename(columns={3: 0, 4: 1, 5: 2, 9: 3}, inplace=False)

# Finding height of torso
maxy = max(data3D)
miny = min(data3D)
xzplane01 = Plane(Point3D(0, maxy, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
xzplane02 = Plane(Point3D(0, miny, 0), Point3D(1, 0, 0), Point3D(0, 0, 1))
