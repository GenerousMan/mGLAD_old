# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import numpy as np
from utils import *
from utils import Cal_ProbLoss
from models import mGLAD
import yaml

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'mGLAD', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
#flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
#flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('ability_num', 300, 'Weight for L2 loss on embedding matrix.')
#flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
#flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
#shape,worker_num,task_num,edge_type,edges,true_labels = read_BlueBirds()
shape,worker_num,task_num,edge_type,edges,true_labels = read_duck()
print(true_labels)
model_func = mGLAD
# Define placeholders
placeholders = {
    'edges': tf.placeholder(tf.int32,shape=shape),
    'worker_num': tf.placeholder(tf.int32),
    'task_num': tf.placeholder(tf.int32),
    'edge_type':tf.placeholder(tf.int32),
    'ability_num':tf.placeholder(tf.int32)
}

# Create model
model = model_func(placeholders, edge_type=edge_type,task_num=task_num,worker_num=worker_num,ability_num=FLAGS.ability_num,input_dim=edges.shape, logging=True)
merged = tf.summary.merge_all()
##initialize

# Initialize session
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
# Define model evaluation function


def evaluate(edges,worker_num,task_num,edge_type,ability_num,placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(edges,worker_num,task_num,edge_type,ability_num,placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(edges,worker_num,task_num,edge_type,FLAGS.ability_num,placeholders)
    #feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,model.outputs,model.layers[0].t,model.layers[0].output,merged], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(edges,worker_num,task_num,edge_type,FLAGS.ability_num,placeholders)
    cost_val.append(cost)
    latent_t_predict=np.argmin(outs[4], axis=1)
    print((np.equal(true_labels,latent_t_predict)).sum()/task_num)
    #print(np.argmax(outs[4],axis=1))
    #print(outs[4])
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    #rs = sess.run(merged)
    writer.add_summary(outs[6], epoch)

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")
'''
# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
'''