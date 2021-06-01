# -------------------------Import Libraries------------------------------#
import numpy as np
import pands as pd
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
path = 'C:/Users/nehal/OneDrive/Desktop/Research ST/'
file = pd.read_csv(path + 'EDM-1.csv')
