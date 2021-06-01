# -------------------------Import Libraries------------------------------#
import numpy as np
import pandas as pd
import open3d as o3
import trimesh
import probreg

clearSet = {"r*", "data*", "plot*", "file*", 'datafile', 'datasize', 'data3D',
  'STDcol', 'numebrofthre', 'directory', 'fnames', 'Numberofolders', 'csvName',
  'csvLength', 'paths'}
clearSet.clear

# -------------------------Open Data Set------------------------------#
Threshold1 = {3, 9.333}
initial = 1
initial1 = 1
numebrofthre = len(Threshold1)

# directory to the EDM file 
# Prompts user to select the folder that the .csv is located in 
path = askdirectory(title='Select Folder') 
datafile = pd.read_csv(path + '/EDM-1.csv', header=None)
# Delete columns
datafile.drop([0, 1, 2, 6, 7, 8], axis=1, inplace=True)
# Rename columns
data3D = datafile.rename(columns={3: 0, 4: 1, 5: 2, 9: 3}, inplace=False)

