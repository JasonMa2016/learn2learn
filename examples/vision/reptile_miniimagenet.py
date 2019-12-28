#!/usr/bin/env python3

import os
import random
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch as th
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

from tensorboardX import SummaryWriter

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = th.zeros(data.size(0)).byte()
    adaptation_indices[th.arange(shots*ways) * 2] = 1
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[1 - adaptation_indices], labels[1 - adaptation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)

    # Evaluate the adapted model
    with th.no_grad():
        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)
        valid_error /= len(evaluation_data)
        valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(args):
    ways = args.ways
    shots = args.shots
    meta_lr = args.meta_lr
    fast_lr = args.fast_lr
    meta_batch_size = args.meta_batch_size
    adaptation_steps = args.adaptation_steps
    num_iterations = args.num_iterations
    cuda = args.cuda
    seed = args.seed

    # logging
    if args.save:
        writer = SummaryWriter('{0}'.format(args.data_dir + '/' + args.output_folder))

        save_folder = '{0}'.format(args.data_dir + '/' + args.output_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # save the configurations
        with open(os.path.join(save_folder, 'config.json'), 'w') as f:
            config = {k: v for (k,v) in vars(args).items() if k != 'device'}
            json.dump(config, f, indent=2)

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    device = th.device('cpu')
    if cuda and th.cuda.device_count():
        print("GPU Activated!")
        th.cuda.manual_seed(seed)
        device = th.device('cuda')

    # Create Datasets
    train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train')
    valid_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='test')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, ways),
        KShots(train_dataset, 2*shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    valid_transforms = [
        NWays(valid_dataset, ways),
        KShots(valid_dataset, 2*shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(train_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)

    test_transforms = [
        NWays(test_dataset, ways),
        KShots(test_dataset, 2*shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        min_accuracy = 1
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)

            min_accuracy = min(min_accuracy, evaluation_accuracy)
            for p1, p2 in zip(learner.parameters(), maml.parameters()):
                p1_copy = p1.clone().detach()
                p2_copy = p2.clone().detach()

                if p2.grad is None:
                    p2.grad = -(p1_copy-p2_copy)
                    # print("really")
                else:
                    p2.grad -= p1_copy-p2_copy

            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # # Compute meta-validation loss
            # learner = maml.clone()
            # batch = valid_tasks.sample()
            # evaluation_error, evaluation_accuracy = fast_adapt(batch,
            #                                                    learner,
            #                                                    loss,
            #                                                    adaptation_steps,
            #                                                    shots,
            #                                                    ways,
            #                                                    device)
            # meta_valid_error += evaluation_error.item()
            # meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()


        training_loss = meta_train_error / meta_batch_size
        training_accuracy = meta_train_accuracy / meta_batch_size
        testing_loss = meta_test_error / meta_batch_size
        testing_accuracy = meta_test_accuracy / meta_batch_size

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', training_loss)
        print('Meta Train Accuracy', training_accuracy)
        # print('Meta Valid Error', meta_valid_error / meta_batch_size)
        # print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
        print('Meta Test Error', testing_loss)
        print('Meta Test Accuracy', testing_accuracy)

        if args.save:
            writer.add_scalar('Loss/training_loss', training_loss, iteration)
            writer.add_scalar('Loss/testing_loss', testing_loss, iteration)
            writer.add_scalar('Accuracy/training_accuracy', training_accuracy, iteration)
            writer.add_scalar('Accuracy/testing_accuracy', testing_accuracy, iteration)
            writer.add_scalar('Accuracy/minimum_accuracy', min_accuracy, iteration)
            # with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
            #     th.save(maml.state_dict(), f)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()



if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Reptile Miniimagenet')

    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--meta-batch-size', type=int, default=32)
    parser.add_argument('--adaptation-steps', type=int, default=5)
    parser.add_argument('--num-iterations', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--meta-lr', type=float, default=0.003)
    parser.add_argument('--fast-lr', type=float, default=0.5)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--save', type=bool, default=True)

    args = parser.parse_args()
    args.data_dir = '../data/'

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    args.output_folder = 'ReptileSeed{}'.format(args.seed)
    main(args)
