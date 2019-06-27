from test import run_with_frozen_pb, display_image
from networks import get_network
import numpy as np
#run_with_frozen_pb("/home/ubuntu/sandbox/hand_labels/manual_train/015374503_01_l.jpg", 192, "/home/ubuntu/landing/PoseEstimationForMobile_v2/training/tttt.pb", "Convolutional_Pose_Machine/stage_5_out")

#run_with_frozen_pb("/home/ubuntu/sandbox/ai_challenger/train/0000e06c1fc586992dc2445e9e102899ccb5e3fc.jpg", 192, "/home/ubuntu/landing/PoseEstimationForMobile_v2/training/xxx.pb", "Convolutional_Pose_Machine/stage_5_out")

#run_with_frozen_pb("/home/ubuntu/sandbox/ai_challenger/train/0000e06c1fc586992dc2445e9e102899ccb5e3fc.jpg", 192, "/home/ubuntu/sandbox/model.pb", "Convolutional_Pose_Machine/stage_5_out")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import cv2

input_w_h = 192

input_node = tf.placeholder(tf.float32, shape=[1, input_w_h, input_w_h, 3], name="image")

img_path = "/home/ubuntu/sandbox/ai_challenger/train/0000e06c1fc586992dc2445e9e102899ccb5e3fc.jpg"
image_0 = cv2.imread(img_path)
w, h, _ = image_0.shape
image_ = cv2.resize(image_0, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)

#checkpoint = "/home/ubuntu/sandbox/trained/mv2_cpm_tiny_3/models/mv2_cpm_batch-16_lr-0.001_gpus-1_192x192_experiments-test_mv2_cpm/model-1690"
checkpoint = "/home/ubuntu/sandbox/trained/mv2_cpm_tiny_4/models/mv2_cpm_batch-16_lr-0.001_gpus-1_192x192_experiments-test_mv2_cpm/model-20700"

with tf.Session() as sess:
    net = get_network("mv2_cpm", input_node, trainable=False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    graph = tf.get_default_graph()
    output_node_names = "Convolutional_Pose_Machine/stage_5_out"
    image = graph.get_tensor_by_name("%s:0" % "image")
    output = graph.get_tensor_by_name("%s:0" % output_node_names)
    print(output)
    heatmaps = sess.run(output, feed_dict={image: [image_]})
    for i in range(14):
        print(i, ":")
        print(np.count_nonzero(heatmaps[0,:,:,i]))
# sess=tf.Session()
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('/home/ubuntu/sandbox/trained/mv2_cpm_tiny_3/models/mv2_cpm_batch-16_lr-0.001_gpus-1_19\
# 2x192_experiments-test_mv2_cpm/model-1690.meta')
# saver.restore(sess,tf.train.latest_checkpoint('/home/ubuntu/sandbox/trained/mv2_cpm_tiny_3/models/mv2_cpm_batch-16_lr-0.001_gpus-1_19\
# 2x192_experiments-test_mv2_cpm/'))



# img_path = "/home/ubuntu/sandbox/ai_challenger/train/0000e06c1fc586992dc2445e9e102899ccb5e3fc.jpg"
# image_0 = cv2.imread(img_path)
# w, h, _ = image_0.shape
# image_ = cv2.resize(image_0, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)

# heatmaps = sess.run(output, feed_dict={
#     "image:0": [image_]})
