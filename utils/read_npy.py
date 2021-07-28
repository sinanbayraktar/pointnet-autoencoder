import numpy as np 
import os 

# PATH = "/home/sinanb/Documents/Master_Thesis/pointnet-autoencoder/logs/log_airplane/preds.npy"
PATH = "/home/sinanb/Documents/Master_Thesis/pointnet-autoencoder/data/OUTPUTS/sampled_vertices_tooth_8.npy"

data = np.load(PATH)


data1 = data[0,:,:]
out_path_temp = os.path.join(os.path.dirname(PATH), "one_out_temp.txt")
out_path = os.path.join(os.path.dirname(PATH), "one_out.obj")

np.savetxt(out_path_temp, data1, delimiter=" ")

f = open(out_path_temp, "r")
f_out = open(out_path, "w")
lines = f.readlines()

for line in lines: 
    f_out.write("v " + line)

f.close()
f_out.close()

os.remove(out_path_temp)

print("THE END")