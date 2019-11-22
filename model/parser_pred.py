# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-mp', '--model_path',
                        type=str,
                        help='Path to the saved torch model.',
                        default='./model.pth')

    parser.add_argument('-pr', '--prototypes',
                        type=str,
                        help='Path to the folder containing data for the cells to be used as prototypes, or .pkl if using cached data.',
                        default='./prototype')
    
    parser.add_argument('-qr', '--queries',
                        type=str,
                        help='Path to the folder containing data for the cells to be used as query, or .pkl if using cached data.',
                        default='./query')

    parser.add_argument('-pf', '--proto_ftype',
                        type=str,
                        help='File format of the data matrix for prototypes, .csv or .mtx',
                        default='.csv')

    parser.add_argument('-qf', '--query_ftype',
                        type=str,
                        help='File format of the data matrix for prototypes, .csv or .mtx',
                        default='.csv')

    parser.add_argument('-pc', '--proto_use_cache',
                        action='store_true',
                        help='Load prototype data from a previously cached .pkl. Provide the path to --prototypes.')

    parser.add_argument('-qc', '--query_use_cache',
                        action='store_true',
                        help='Load query data from a previously cached .pkl. Provide the path to --queries.')

    parser.add_argument('-pcp', '--proto_cache_path',
                        type=str,
                        help='If provided, will cache prototype data to a .pkl at this path.',
                        default='')

    parser.add_argument('-qcp', '--query_cache_path',
                        type=str,
                        help='If provided, will cache query data to a .pkl at this path.',
                        default='')

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='Random seed to use.',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='Use GPU for network execution. Significantly faster!')

    return parser
