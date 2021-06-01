# -------------------------Import Libraries------------------------------#
import numpy as np
import open3d as o3
import trimesh
import probreg

clearSet = {"r*", "data*", "plot*", "file*", 'datafile', 'datasize', 'data3D',
  'STDcol', 'numebrofthre', 'directory', 'fnames', 'Numberofolders', 'csvName',
  'csvLength', 'paths'}
clearSet.clear

# 2. Be able to open data set
Threshold1 = {3, 9.333}
initial = 1
initial1 = 1
numebrofthre = len(Threshold1)

#directory to the EDM file 
path = 'C:/Users/nehal/OneDrive/Desktop/Research ST'
file = np.loadtxt(path + "EDM-1.csv")

# directory = SystemDialogInput["Directory"]
# fn1 = FileNames[] numberoffiles=Length[fn1]
# if directory == Canceled, SetDirectory(directory)


# paths = StringTake[fnames[[#]], pathLength[[#]] - csvLength] & /@ 
#   Range@Numberofolders;(*directory\ of each folder*)
# Print["Folder order to check=" <> ToString@paths];
# ProgressIndicator[Dynamic[countfile/Numberofolders]]
