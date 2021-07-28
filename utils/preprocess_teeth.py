import numpy as np 
import os 
import glob
from numpy.random import shuffle 
import trimesh
import sampling


## Split dataset into train/val/test
# Taken from SampleNet/reconstruction/in_out.py and edited
def split_data(data, split, shuffle=False, perm=None):
    assert sum(split) == 1.0, "data split does not sum to 1: %.2f" % sum(split)

    num_examples = data.shape[0]
    if perm is not None:
        assert perm.shape[0] == data.shape[0], "perm.shape: %s data.shape: %s" % (
            perm.shape,
            data.shape,
        )
    else:
        perm = np.arange(num_examples)
        if shuffle:
            np.random.shuffle(perm)

    data = data[perm]
    train_end = int(round(split[0] * num_examples))
    val_end = int(round((split[0] + split[1]) * num_examples))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    counts = [train_data.shape[0], val_data.shape[0], test_data.shape[0]]
    assert sum(counts) == num_examples, (
        "data split (%d, %d, %d) does not sum to num_examples (%d)"
        % (counts[0], counts[1], counts[2], num_examples)
    )

    return train_data, val_data, test_data, perm


## Parameters 
NUM_POINTS = 2048
DATA_SPLIT = (0.85, 0.05, 0.10)
APPEND_DATA = 0

## Arange paths 
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data/teeth_scan_data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/OUTPUTS")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
if not os.path.exists(VAL_DIR):
    os.mkdir(VAL_DIR)
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

teeth_folders = os.listdir(DATA_DIR)
print("There are ", len(teeth_folders), " teeth scan folders")


## Create output numpy array lists in the dictionary
sampled_vertices_lists = {}
sampled_vertices_center_lists = {}
for i in range(1,35):
    sampled_vertices_lists[i] = []
    sampled_vertices_center_lists[i] = []


## Loop over the folders
for folder_name in teeth_folders:
    teeth_folder = os.path.join(DATA_DIR, folder_name)
    # Loop over the files 
    file_list = sorted(glob.glob(os.path.join(teeth_folder, "*.stl")))
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        if file_name == "upper.stl" or file_name == "lower.stl":
            continue
        elif file_name[:5] == "tooth":
            tooth_id = int(file_name[file_name.find("_")+1:file_name.find(".")])
        elif file_name == "upper_g.stl":
            tooth_id = 33
        elif file_name == "lower_g.stl":
            tooth_id = 34
        else: 
            print("Wrong file name: ", file_name)
        
        mesh = trimesh.load(file_path)
        mesh_vertices = np.array(mesh.vertices)
        sampled_vertex_indices, sampled_vertex_distances = sampling.farthest_point_sampling(mesh_vertices, NUM_POINTS)
        sampled_mesh_vertices =  mesh_vertices[sampled_vertex_indices]
        sampled_mesh_vertices = np.squeeze(sampled_mesh_vertices, axis=0)

        sampled_vertices_lists[tooth_id].append(sampled_mesh_vertices)
        sampled_vertices_center_lists[tooth_id].append(mesh.center_mass)


for i in range(1,34):
    curr_vert_arr = np.array(sampled_vertices_lists[i])
    curr_vert_cent_arr = np.array(sampled_vertices_center_lists[i])

    appended_vert = curr_vert_arr
    appended_cent = curr_vert_cent_arr
    for k in range(APPEND_DATA-1):
        appended_vert = np.concatenate((appended_vert, curr_vert_arr), axis=0)
        appended_cent = np.concatenate((appended_cent, curr_vert_cent_arr), axis=0)

    ## Split into train/val/test splits 
    train, val, test, perm = split_data(appended_vert, DATA_SPLIT, shuffle=False, perm=None)
    train_c, val_c, test_c, _ = split_data(appended_cent, DATA_SPLIT, shuffle=False, perm=perm)

    ## Save npy files in the output directory 
    out_vert_name = "sampled_vertices_tooth_" + str(i) + "_train"
    np.save(os.path.join(TRAIN_DIR, out_vert_name), train)
    out_vert_name = "sampled_vertices_tooth_" + str(i) + "_val"
    np.save(os.path.join(VAL_DIR, out_vert_name), val)
    out_vert_name = "sampled_vertices_tooth_" + str(i) + "_test"
    np.save(os.path.join(TEST_DIR, out_vert_name), test)

    out_cent_name = "center_tooth_" + str(i) + "_train"
    np.save(os.path.join(TRAIN_DIR, out_cent_name), train_c)
    out_cent_name = "center_tooth_" + str(i) + "_val"
    np.save(os.path.join(VAL_DIR, out_cent_name), val_c)
    out_cent_name = "center_tooth_" + str(i) + "_test"
    np.save(os.path.join(TEST_DIR, out_cent_name), test_c)


print("THE END")

