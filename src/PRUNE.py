# -*- coding: utf-8 -*-
# Reference implementation of "PRUNE"
# Author: Yi-An Lai
# PRUNE: Preserving Proximity and Global Ranking for Network Embedding
# Yi-An Lai*, Chin-Chi Hsu*, Wen-Hao Chen, Ming-Han Feng, and Shou-De Lin
# Advances in Neural Information Processing Systems (NIPS), 2017

import numpy as np
import tensorflow as tf


def forward_pass(featX, scope, n_emb, n_latent):
    """
    PRUNE's generative process for each node
    Generate ranking score and proximity representation
    """
    with tf.variable_scope(scope):
        # hidden layers dimension
        n_hidden_rank = 128
        n_hidden_prox = 128

        # global node ranking score
        W_hr = tf.get_variable("W_hidden_rank", [n_emb, n_hidden_rank])
        b_hr = tf.get_variable("b_hidden_rank", [n_hidden_rank])
        layer_rank = tf.nn.elu(tf.add(tf.matmul(featX, W_hr), b_hr))

        W_or = tf.get_variable("W_output_rank", [n_hidden_rank, 1])
        b_or = tf.get_variable("b_output_rank", [1])
        rank_pi = tf.nn.softplus(tf.add(tf.matmul(layer_rank, W_or), b_or))

        # proximity representation
        W_hp = tf.get_variable("W_hidden_prox", [n_emb, n_hidden_prox])
        b_hp = tf.get_variable("b_hidden_prox", [n_hidden_prox])
        layer_prox = tf.nn.elu(tf.add(tf.matmul(featX, W_hp), b_hp))

        W_op = tf.get_variable("W_output_prox", [n_hidden_prox, n_latent])
        b_op = tf.get_variable("b_output_prox", [n_latent])
        prox_rep = tf.nn.relu(tf.add(tf.matmul(layer_prox, W_op), b_op))

    return rank_pi, prox_rep


def initialize_PRUNE(scope_name, n_emb, learning_rate, nodeCount, lamb):
    """
    Initialize PRUNE
    """
    # proximity representation dimension
    n_latent = 64

    # indegree and outdegree
    outdeg = tf.placeholder("float", [None])
    indeg = tf.placeholder("float", [None])

    # indexes for head and tail nodes, PMI values
    node_heads = tf.placeholder(tf.int32, [None])
    node_tails = tf.placeholder(tf.int32, [None])
    pmis = tf.placeholder("float", [None, 1])

    with tf.variable_scope(scope_name) as scope:
        # Create parameters
        initializer = tf.contrib.layers.xavier_initializer()
        tf.get_variable_scope().set_initializer(initializer)

        embeddings = tf.get_variable("emb", [nodeCount, n_emb])
        heads_emb = tf.gather(embeddings, node_heads)
        tails_emb = tf.gather(embeddings, node_tails)

        # W_shared
        W_init = np.identity(n_latent)
        W_init += abs(np.random.randn(n_latent, n_latent) / 1000.0)
        W_initializer = tf.constant_initializer(W_init)
        W_shared = tf.get_variable("W_shared", [n_latent, n_latent],
                                   initializer=W_initializer)
        W_shared_posi = tf.nn.relu(W_shared)

        # forward pass for head nodes
        heads_pi, heads_prox = forward_pass(heads_emb, scope, n_emb, n_latent)

        # Siamese neural network: reuse the NN defined
        tf.get_variable_scope().reuse_variables()

        # forward pass for tail nodes
        tails_pi, tails_prox = forward_pass(tails_emb, scope, n_emb, n_latent)

        zWz = heads_prox * tf.matmul(tails_prox, W_shared_posi)
        zWz = tf.reduce_sum(zWz, 1, keep_dims=True)

        # preserving proximity
        prox_loss = tf.reduce_mean((zWz - pmis)**2)

        # preserving global ranking
        r_loss = indeg * (tf.square(-tails_pi / indeg + heads_pi / outdeg))
        ranking_loss = tf.reduce_mean(r_loss)

        # define costs
        cost = prox_loss + lamb * ranking_loss

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    inits = (init, optimizer, cost, heads_pi, heads_prox, tails_pi, tails_prox,
             W_shared, node_heads, node_tails, indeg, outdeg, pmis, embeddings)
    return inits


def train_PRUNE(init, optimizer, cost, node_heads, node_tails,
                indeg, outdeg, pmis, embeddings, scope_name, epoch,
                graph, PMI_values, batchsize, out_degrees, in_degrees,
                gpu_fraction=0.20, print_every_epoch=3, save_cp=False):
    """
    PRUNE training process
    """
    # set 0 values to 1 to avoid divided by zero
    out_degrees[out_degrees == 0] = 1
    in_degrees[in_degrees == 0] = 1

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    with tf.variable_scope(scope_name, reuse=True) as scope:  # noqa

        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.InteractiveSession(config=tf_config)
        with sess.as_default():

            # initialize variables
            sess.run(init)
            vars_to_train = tf.trainable_variables()
            saver = tf.train.Saver(vars_to_train)

            # Training epochs
            for n in range(epoch):
                obj = 0.0
                indexes = np.random.permutation(len(graph))

                # Update parameters with mini-batch data
                num_mini_batch = int(len(graph) / batchsize)
                for i in range(num_mini_batch):
                    inds = indexes[i * batchsize: (i + 1) * batchsize]
                    edges = graph[inds]
                    pmi_vals = PMI_values[inds]

                    feed_dict = {
                        outdeg: out_degrees[edges[:, 0]],
                        indeg: in_degrees[edges[:, 1]],
                        pmis: pmi_vals,
                        node_heads: edges[:, 0].astype(np.int32),
                        node_tails: edges[:, 1].astype(np.int32)
                    }

                    _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
                    obj += c

                # Information printing
                if print_every_epoch and ((n + 1) % print_every_epoch == 0):
                    print('\tepoch: %d; obj: %.7f' %
                          (n + 1, obj / num_mini_batch))

                # TensorFlow checkpoints
                if save_cp and (n + 1) % 10 == 0:
                    saver.save(sess, './checkpoints/PRUNE.ckpt',
                               global_step=(n + 1))

            final_embeddings = sess.run(embeddings)

    return final_embeddings


def compute_PMI(graph, nodeCount, in_degrees, out_degrees, alpha=5.0):
    """
    Compute pointwise mutual information (PMI) for each edge
    alpha: reprensents the number of effective negative samples
           for each positive sample
    """
    PMI_values = np.zeros((len(graph), 1))
    for ind in range(len(graph)):
        head, tail = graph[ind]
        pmi = len(graph) / alpha / out_degrees[head] / in_degrees[tail]
        PMI_values[ind, 0] = np.log(pmi)

    PMI_values[PMI_values < 0] = 0

    return PMI_values


def run_PRUNE(lamb, graph, nodeCount, n_emb, learning_rate, epoch,
              gpu_fraction=0.20, batchsize=1024, print_every_epoch=1,
              scope_name='default', save_cp=False):
    """
    Compute indegrees, outdegrees, PMI
    Initialize and train PRUNE for node embeddings
    """
    # compute indegrees, outdegrees, PMI values
    out_degrees = np.zeros(nodeCount)
    in_degrees = np.zeros(nodeCount)
    for node_i, node_j in graph:
        out_degrees[node_i] += 1
        in_degrees[node_j] += 1

    PMI_values = compute_PMI(graph, nodeCount, in_degrees, out_degrees)

    # initialize PRUNE
    inits = initialize_PRUNE(scope_name, n_emb, learning_rate,
                             nodeCount, lamb)

    (init, optimizer, cost, heads_pi, heads_prox, tails_pi, tails_prox,
     W_shared, node_heads, node_tails, indeg, outdeg, pmis, embeddings) = inits

    # train PRUNE
    embeddings = train_PRUNE(init, optimizer, cost, node_heads,
                             node_tails, indeg, outdeg, pmis, embeddings,
                             scope_name, epoch, graph, PMI_values,
                             batchsize, out_degrees, in_degrees,
                             gpu_fraction=gpu_fraction,
                             print_every_epoch=print_every_epoch,
                             save_cp=save_cp)

    return embeddings
