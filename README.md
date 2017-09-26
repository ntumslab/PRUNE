# TriFac_Rank


# TriFac-Rank
[![Build Status](https://travis-ci.org/ntumslab/TriFac_Rank.svg?branch=master)](https://travis-ci.org/ntumslab/TriFac_Rank)

TriFac-Rank is an unsupervised generative approach for network embedding. The approach entails a multi-task Siamese neural network to connect embedding vectors and our objective to preserve the global node ranking and local proximity of nodes. Deeper analysis for the proposed architecture and objective can be found in the paper.

TriFac-Rank satisfies the following design properties: scalability, asymmetry, unity and simplicity.

A reference implementation of **TriFac-Rank** in the paper (please see the file - TriFac_Rank_nips_preprint.pdf):<br>
> Preserving Proximity and Global Ranking for Node Embedding<br>
> Yi-An Lai+, Chin-Chi Hsu+, Wen-Hao Chen, Ming-Han Feng, and Shou-De Lin<br>
> Advances in Neural Information Processing Systems (NIPS), 2017 <br>
> +: These authors contributed equally to this paper.

## Usage

### Example
Run TriFac-Rank on the sample graph:

    python src/main.py --inputgraph sample/graph.edgelist

#### Options
Check out optional arguments such as learning rate, epochs, GPU usage by:

    python src/main.py --help

### Inputs
Supported graph format is the edgelist:

    node_from node_to

Input graph are treated as directed.

### Output

A comma-separated table of embeddings, the k-th row represents the k-th node's embeddings:

    embed_dim1, embed_dim2, ...

## Requirements
Install all dependencies:

    pip install -r requirements.txt

This implementation is built on `tensorflow 1.1.0`. If using Mac OS or encountering other problems, see detailed TensorFlow installing guild at: 
[https://www.tensorflow.org/versions/r1.1/install/](https://www.tensorflow.org/versions/r1.1/install/)

## Citing

If you find **TriFac-Rank** useful in your research, please consider citing the paper:

    Preserving Proximity and Global Ranking for Node Embedding, NIPS 2017.

## Miscellaneous

If having any questions, please contact us at Yi-An Lai <b99202031@ntu.edu.tw> or Chin-Chi Hsu <chinchi@iis.sinica.edu.tw>.
