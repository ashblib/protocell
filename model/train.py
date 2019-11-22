# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from singlecell_dataset import TrainingDataset
from protonet import ProtoNetBig
from parser_train import get_parser
from misc_utils import fancy_text as fncy_txt

from tqdm import tqdm
import numpy as np
import torch
import os
import sys


def init_seed(opt):
    '''
    Initialize random seeds. Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode, iteration):
    '''
    Initialize the dataset
    '''
    dataset = TrainingDataset(root=opt.root, ftype=opt.file_type, mode=mode, train_frac=opt.train_frac, cv_iter=iteration)
    if mode == "train":
        print("Training classes:")
    print(dataset.class_map)
    return dataset, dataset.x[0].shape[0]


def init_sampler(opt, labels, mode, n_proto_support):
    '''
    Initialize the sampler
    '''
    if 'train' in mode:
        classes, counts = np.unique(labels, return_counts=True)
        classes_per_it = len(classes)
        num_samples = n_proto_support + opt.num_query_tr
        for idx, count in enumerate(counts):
            if num_samples > count:
                print("*** Error ***: You do not have enough samples in class {} for training -- {} samples vs {} needed.".format(idx, count, num_samples))
                sys.exit()
    else:
        classes, counts = np.unique(labels, return_counts=True)
        classes_per_it = len(classes)
        num_samples = n_proto_support + opt.num_query_val
        for idx, count in enumerate(counts):
            if num_samples > count:
                print("*** Error ***: You do not have enough samples in class {} for testing -- {} samples vs {} needed.".format(idx, count, num_samples))
                sys.exit()
    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode, iteration, n_proto_support):
    '''
    Initialize the dataloader
    '''
    dataset, n_features = init_dataset(opt, mode, iteration)
    sampler = init_sampler(opt, dataset.y, mode, n_proto_support)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader, n_features


def init_protonet(opt, n_feat):
    '''
    Initialize the ProtoNet model
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNetBig(x_dim=n_feat).to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, test_dataloader=None, iteration=0, n_proto_support=1):
    '''
    Main training loop for a single cross validation iteration.

    Args:
        opt (options): Options object returned by arg parser.
        tr_dataloader (data.Dataloader): Pytorch dataloader for training data.
        model (ProtoNet): The pytorch nn model.
        optim (torch.optim): Pytorch optimizer class
        lr_scheduler (optim.lr_scheduler): Learning rate scheduler class
        test_dataloader (data.Dataloader): Pytorch dataloader for testing data
        iteration (int): Cross validation iteration (used as random seed)
        n_proto_support (int): Number of support examples per class to train with

    Returns:
        best_state (dict): Model state for best test accuracy
        best_acc (float): Best test accuracy 
        train_loss (float): Last avg training loss
        train_acc (float): Last training accuracy
        val_loss (float): Last testing loss
        val_acc (float): Last testing accuracy
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if test_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model_{}_shot_{}.pth'.format(n_proto_support, iteration))
    last_model_path = os.path.join(opt.experiment_root, 'last_model_{}_shot_{}.pth'.format(n_proto_support, iteration))

    for epoch in range(opt.epochs):
        print(fncy_txt('Epoch {}'.format(epoch)))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc, _ = loss_fn(model_output, target=y,
                                n_support=n_proto_support)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if test_dataloader is None:
            continue
        test_iter = iter(test_dataloader)
        model.eval()
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc, _ = loss_fn(model_output, target=y,
                                n_support=n_proto_support)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Test Loss: {}, Avg Test Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc

def main():
    '''
    Trains model n_ensemble times on different data splits, using num_proto_min to 
    num_proto_max prototypes. This will train n_ensemble*(num_proto_max-num_proto_min) models... be careful
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    n_iterations = options.n_ensemble

    for n_proto_support in range(options.num_proto_min, options.num_proto_max+1):
        avg_test_acc = 0
        avg_val_acc = 0

        init_seed(options)
        for iteration in range(n_iterations):
            tr_dataloader, _ = init_dataloader(options, 'train', iteration, n_proto_support)
            test_dataloader, n_features = init_dataloader(options, 'test', iteration, n_proto_support)
            
            model = init_protonet(options, n_features)
            optim = init_optim(options, model)
            lr_scheduler = init_lr_scheduler(options, optim)
            res = train(opt=options,
                        tr_dataloader=tr_dataloader,
                        test_dataloader=test_dataloader,
                        model=model,
                        optim=optim,
                        lr_scheduler=lr_scheduler,
                        iteration=iteration,
                        n_proto_support=n_proto_support)

if __name__ == '__main__':
    main()
