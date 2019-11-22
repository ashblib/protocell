# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--root',
                        type=str,
                        help='path to dataset',
                        default='..' + os.sep + 'dataset')

    parser.add_argument('-ftype', '--file_type',
                        type=str,
                        help='File type for data',
                        default='..' + os.sep + 'dataset')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=10)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.0001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=10)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=10)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=5)

    parser.add_argument('-nens', '--n_ensemble',
                        type=int,
                        help='Number of networks to train for ensemble',
                        default=1)

    parser.add_argument('-npmin', '--num_proto_min',
                        type=int,
                        help='Min number of prototypes to train network for',
                        default=5)

    parser.add_argument('-npmax', '--num_proto_max',
                        type=int,
                        help='Max number of prototypes to train network for',
                        default=5)

    parser.add_argument('-ts', '--test_split',
                        type=int,
                        help='How many classes to hold out for testing/validation',
                        default=10)

    parser.add_argument('-tf', '--train_frac',
                        type=float,
                        help='Fraction of cells per class to use for training. The rest are used for testing',
                        default=0.8)

    parser.add_argument('-ms', '--min_samples',
                        type=int,
                        help='Filters out classes with fewer than this number of samples',
                        default=20)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    return parser
