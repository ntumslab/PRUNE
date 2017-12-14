"""
Implementation of PRUNE.
Author: Yi-An Lai
For more details, refer to the paper:
PRUNE: Preserving Proximity and Global Ranking for Network Embedding
Yi-An Lai*, Chin-Chi Hsu*, Wen-Hao Chen, Ming-Han Feng, and Shou-De Lin
Advances in Neural Information Processing Systems (NIPS), 2017
"""

import argparse
import numpy as np

from PRUNE import run_PRUNE


def parse_args():
    '''
    Parses PRUNE arguments.
    '''
    parser = argparse.ArgumentParser(description="Run PRUNE.")

    parser.add_argument('--inputgraph', nargs='?',
                        default='sample/graph.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='graph.embeddings',
                        help='Output node embeddings of the graph')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Embedding dimension. Default is 128.')

    parser.add_argument('--lamb', type=float, default=0.01,
                        help='Parameter lambda in objective. Default is 0.01.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for Adam. Default is 1e-4.')

    parser.add_argument('--epoch', type=int, default=50,
                        help='Training epochs. Default is 50.')

    parser.add_argument('--gpu_fraction', type=float, default=0.20,
                        help='Memory usage of the GPU. Default is 0.20.')

    parser.add_argument('--batchsize', type=int, default=1024,
                        help='Size of one mini-batch. Default is 1024.')

    parser.add_argument('--print_every_epoch', type=int, default=1,
                        help='Print the objective every k epochs. Default: 1.')

    parser.add_argument('--save_checkpoints', dest='save_checkpoints',
                        action='store_true',
                        help='Save checkpoints when training. Default: False.')
    parser.set_defaults(save_checkpoints=False)

    return parser.parse_args()


def main(args):
    """
    Pipeline for unsupervised node embeddings
    preserving proximity and global ranking properties
    """
    # Read the graph into numpy array
    graph = np.loadtxt(args.inputgraph).astype(np.int32)
    nodeCount = graph.max() + 1

    # PRUNE: node embeddings
    embeddings = run_PRUNE(args.lamb, graph, nodeCount,
                           args.dimension, args.learning_rate,
                           args.epoch, args.gpu_fraction,
                           args.batchsize, args.print_every_epoch,
                           save_cp=args.save_checkpoints)
    np.savetxt(args.output, embeddings, delimiter=',')


if __name__ == "__main__":
    args = parse_args()
    main(args)
