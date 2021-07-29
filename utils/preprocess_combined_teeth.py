import numpy as np 
import os 
import glob
from numpy.random import shuffle 
import pymeshlab 
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


## Create train/val/test permutations for dataset
def get_split_perm(data_size, split, shuffle=False):
    assert sum(split) == 1.0, "data split does not sum to 1: %.2f" % sum(split)

    perm = np.arange(data_size)
    if shuffle:
        np.random.shuffle(perm)

    train_end = int(round(split[0] * data_size))
    val_end = int(round((split[0] + split[1]) * data_size))

    train_perm = perm[:train_end]
    val_perm = perm[train_end:val_end]
    test_perm = perm[val_end:]
    
    counts = [train_perm.shape[0], val_perm.shape[0], test_perm.shape[0]]
    assert sum(counts) == data_size, (
        "data split (%d, %d, %d) does not sum to data_size (%d)"
        % (counts[0], counts[1], counts[2], data_size)
    )
    
    return train_perm, val_perm, test_perm


## Uses the result of get_split_perm() function
def apply_perm_to_data(data, train_perm, val_perm, test_perm):
    train_data = data[train_perm]
    val_data = data[val_perm]
    test_data = data[test_perm]

    return train_data, val_data, test_data


## Combine meshes together to a one mesh 
def create_combined_mesh(paths, out_path=None):
    ms = pymeshlab.MeshSet()
    for i in range(len(paths)):
        ## Read mesh 
        ms.load_new_mesh(paths[i])

    ms.flatten_visible_layers()
    assert ms.number_meshes() == 1, "Flatenning is failed!"
    out_mesh = ms.current_mesh()
    vertices = out_mesh.vertex_matrix()
    ms.clear()

    return vertices


## Parameters 
NUM_POINTS = 2048
DATA_SPLIT = (0.85, 0.05, 0.10)
APPEND_DATA = 10

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
folder_num = len(teeth_folders)
print("There are ", folder_num, " teeth scan folders")

## Create output array 
out_array_upper = np.zeros((folder_num, NUM_POINTS, 3))
out_array_lower = np.zeros((folder_num, NUM_POINTS, 3))

## Loop over the folders
for i, folder_name in enumerate(teeth_folders):
    print("Folder ", folder_name)
    teeth_folder_path = os.path.join(DATA_DIR, folder_name)
    # Create upper & lower tooth path lists
    upper_teeth = []
    lower_teeth = []
    
    ## Loop over the files 
    file_list = sorted(glob.glob(os.path.join(teeth_folder_path, "*.stl")))
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
        
        ## Put tooth path into the list
        if tooth_id <= 16: # upper
            upper_teeth.append(file_path)
        elif 17 <= tooth_id and tooth_id <= 32: # lower
            lower_teeth.append(file_path)

    ## Combine meshes 
    vertices_upper = create_combined_mesh(upper_teeth)
    vertices_lower = create_combined_mesh(lower_teeth)

    ## Sample mesh
    # upper
    upper_sampled_vertex_indices, upper_sampled_vertex_distances = sampling.farthest_point_sampling(vertices_upper, NUM_POINTS)
    upper_sampled_mesh_vertices =  vertices_upper[upper_sampled_vertex_indices]
    upper_sampled_mesh_vertices = np.squeeze(upper_sampled_mesh_vertices, axis=0)
    out_array_upper[i,:,:] = upper_sampled_mesh_vertices
    # lower
    lower_sampled_vertex_indices, lower_sampled_vertex_distances = sampling.farthest_point_sampling(vertices_lower, NUM_POINTS)
    lower_sampled_mesh_vertices =  vertices_lower[lower_sampled_vertex_indices]
    lower_sampled_mesh_vertices = np.squeeze(lower_sampled_mesh_vertices, axis=0)
    out_array_lower[i,:,:] = lower_sampled_mesh_vertices


## Split data
train_perm, val_perm, test_perm = get_split_perm(folder_num, DATA_SPLIT, shuffle=True)
# upper
train_data_upper, val_data_upper, test_data_upper = apply_perm_to_data(out_array_upper, train_perm, val_perm, test_perm)
# lower
train_data_lower, val_data_lower, test_data_lower = apply_perm_to_data(out_array_lower, train_perm, val_perm, test_perm)


## Save output npy array files 
# upper
np.save(os.path.join(TRAIN_DIR, "train_upper"), train_data_upper)
np.save(os.path.join(VAL_DIR, "val_upper"), val_data_upper)
np.save(os.path.join(TEST_DIR, "test_upper"), test_data_upper)
# lower
np.save(os.path.join(TRAIN_DIR, "train_lower"), train_data_lower)
np.save(os.path.join(VAL_DIR, "val_lower"), val_data_lower)
np.save(os.path.join(TEST_DIR, "test_lower"), test_data_lower)


print("THE END")

