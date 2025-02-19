import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import show3d_balls
import part_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="shapenetcore", help="teeth or shapenetcore dataset [default: shapenetcore]")
parser.add_argument('--tooth_id', default="8", help="Tooth class: 1-32 for tooth, 33-34 for upper/lower gums [default: 8]")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--num_group', type=int, default=1, help='Number of groups of generated points -- used for hierarchical FC decoder. [default: 1]')
FLAGS = parser.parse_args()


MODEL_PATH = FLAGS.model_path
EVAL_DIR = os.path.dirname(MODEL_PATH)
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model) # import network module

if FLAGS.dataset == "shapenetcore":
    DATA_PATH = os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0')
    TEST_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=FLAGS.category, split='test',normalize=True)
elif FLAGS.dataset == "teeth":
    DATA_PATH = os.path.join(BASE_DIR, 'data/teeth_split_data')
    TEST_DATASET = part_dataset.TeethDataset(root=DATA_PATH, tooth_id=FLAGS.tooth_id, split="test")
DATASET_SIZE = len(TEST_DATASET)
print "The length of the test dataset is ", DATASET_SIZE


def get_model(batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}
        return sess, ops

def inference(sess, ops, pc, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], 3))
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                     ops['is_training_pl']: False}
        batch_logits = sess.run(ops['pred'], feed_dict=feed_dict)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
    return logits



if __name__=='__main__':

    num_group = FLAGS.num_group
    if num_group > 1:
        color_list = []
        for i in range(num_group):
            color_list.append(np.random.random((3,)))

    sess, ops = get_model(batch_size=1, num_point=NUM_POINT)
    indices = np.arange(DATASET_SIZE)
    np.random.shuffle(indices)
    preds = np.empty((DATASET_SIZE, NUM_POINT, 3), dtype="float32")
    stop_visualization = False
    
    for i in range(DATASET_SIZE):
        if TEST_DATASET.dataset == "teeth":
            ps = TEST_DATASET[indices[i]]
        elif TEST_DATASET.dataset == "shapenetcore":
            ps, seg = TEST_DATASET[indices[i]]
        pred = inference(sess, ops, np.expand_dims(ps,0), batch_size=1) 
        pred = pred.squeeze()
        preds[i, :, :] = pred

        if not stop_visualization: 
            cmd = show3d_balls.showpoints(ps, ballradius=8)
            if cmd == 27: # ESC
                stop_visualization = True 
        if not stop_visualization: 
            cmd = show3d_balls.showpoints(pred, ballradius=8, gradient=True)
            if cmd == 27: # ESC
                stop_visualization = True 

        if num_group > 1:
            c_gt = np.zeros_like(pred)
            for i in range(num_group):
                c_gt[i*NUM_POINT/num_group:(i+1)*NUM_POINT/num_group,:] = color_list[i]
            show3d_balls.showpoints(pred, c_gt=c_gt, ballradius=8)

    # Save all predictions to a npy file 
    preds_path = os.path.join(EVAL_DIR, "preds")
    np.save(preds_path, preds)


    print("END OF TESTING")