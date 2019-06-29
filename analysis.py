from __future__ import division
from __future__ import print_function

from operator import itemgetter
from process_data import DecagonData
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics
from sklearn import metrics

import numpy as np
import tensorflow as tf
import pickle
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ""


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert decagon.adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert decagon.adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


# NUM_EDGE = 1317
# et = [i for i in range(NUM_EDGE)] + [i for i in range(NUM_EDGE)]         # ordered edge types
# data = DecagonData(et)
# data = data.adj_mats_orig[1, 1]

# ########################### For Embedding Check ########################## #
# adj = data.adj_mats_orig
# adj[1, 1][0].diagonal().max()


# ########################### Histogram of DD Edge Type ########################## #
# tmp = [data[i].nnz for i in range(NUM_EDGE)]
# tmp = np.sort(tmp)
#
# n = 57      #
# num = plt.hist(tmp, bins=57)
# plt.xlabel('Number of Times A D-D Edge Type Occurs')
# plt.ylabel('Numbers of D-D Edge Type')
# plt.title('Frequency Distribution Histogram of D-D Edge Type')
# plt.grid()
# # plt.yscale('log')
# plt.savefig('hist_dd_edge.png')
# plt.show()


# ########################### Sample training DD edge types ########################## #
# tmp = np.array([data[i].nnz for i in range(NUM_EDGE)])
#
# lower_bound = 500
# higher_bound = 100000
#
# boolean = np.logical_and(tmp > lower_bound, tmp <= higher_bound)
# indices = np.nonzero(boolean)[0].tolist()
#
# with open("./data_decagon/training_samples_500.pkl", "wb") as f:
#     pickle.dump(indices, f)


# ########################### Load check point and analysis ########################## #
# load selected training
with open("./data_decagon/training_samples_500.pkl", "rb") as f:
    et = pickle.load(f)
et += et
print("The training edge types are: ", et)
print("Total ", int(len(et)/2), " DD edge types have been trained...")

decagon = DecagonData(et)
val_test_size = 0.1

# Settings and placeholders
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')

print("Defining placeholders")
placeholders = construct_placeholders(decagon.edge_types)

# Create minibatch iterator, model and optimizer
minibatch = EdgeMinibatchIterator(
    adj_mats=decagon.adj_mats_orig,
    feat=decagon.feat,
    edge_types=decagon.edge_types,
    et=et,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size,
)

model = DecagonModel(
    placeholders=placeholders,
    num_feat=decagon.num_feat,
    nonzero_feat=decagon.num_nonzero_feat,
    edge_types=decagon.edge_types,
    decoders=decagon.edge_type2decoder,
    name='bw12',
    logging='logging'
)

with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=decagon.degrees,
        edge_types=decagon.edge_types,
        edge_type2dim=decagon.edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}

# restore all the variables.
saver = tf.train.Saver()
# Restore variables from disk.
saver.restore(sess, "./tmp/model.ckpt")

# feed to tf dictionary
while not minibatch.end():
    # Construct feed dictionary
    feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
    feed_dict = minibatch.update_feed_dict(
        feed_dict=feed_dict,
        dropout=FLAGS.dropout,
        placeholders=placeholders)

# score-based evaluation
score = []
for et in range(decagon.num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    score.append(roc_score)
    score.append(auprc_score)
    score.append(apk_score)
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
    print()

# save scores to file
with open("./analysis/result_decagon.pkl", "wb") as f:
    pickle.dump(score, f)

