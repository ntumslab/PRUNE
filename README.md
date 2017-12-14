# PRUNE
[![Build Status](https://travis-ci.org/ntumslab/PRUNE.svg?branch=master)](https://travis-ci.org/ntumslab/PRUNE)

**PRUNE** is an unsupervised generative approach for network embedding.

Design properties **PRUNE** satisfies: scalability, asymmetry, unity and simplicity.

The approach entails a multi-task Siamese neural network to connect embeddings and our objective, preserving global node ranking and local proximity of nodes.

Deeper analysis for the proposed architecture and objective can be found in the paper (please see - *[PRUNE](https://nips.cc/Conferences/2017/Schedule?showEvent=9301)*): <br>
> PRUNE: Preserving Proximity and Global Ranking for Network Embedding<br>
> Yi-An Lai+, Chin-Chi Hsu+, Wen-Hao Chen, Ming-Han Feng, and Shou-De Lin<br>
> Advances in Neural Information Processing Systems (NIPS), 2017 <br>
> +: These authors contributed equally to this paper.

This repo contains reference implementation of **PRUNE**.

## Usage

### Example
Run **PRUNE** on the sample graph:

    python src/main.py --inputgraph sample/graph.edgelist

#### Options
Check out optional arguments such as *learning rate*, *epochs*, *GPU usage* by:

    python src/main.py --help

### Input
Supported graph format is the edgelist:

    node_from node_to

Input graph are treated as directed.

### Output

A comma-separated table of embeddings, the k-th row represents the k-th node's embeddings:

    node_0  embed_dim1, embed_dim2, ...
    node_1  embed_dim1, embed_dim2, ...
            ...

## Requirements
Install all dependencies:

    pip install -r requirements.txt

This implementation is built on `tensorflow 1.1.0`. If using Mac OS or encountering other problems, see detailed TensorFlow installation guild at: 
[https://www.tensorflow.org/versions/r1.1/install/](https://www.tensorflow.org/versions/r1.1/install/)

## Citing

If you find **PRUNE** useful in your research, please consider citing the paper:

    PRUNE: Preserving Proximity and Global Ranking for Network Embedding, NIPS 2017.

## Miscellaneous

If having any questions, please contact us: Yi-An Lai (<b99202031@ntu.edu.tw>) or Chin-Chi Hsu (<chinchi@iis.sinica.edu.tw>). 
