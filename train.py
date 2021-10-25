import os
import time

import pandas as pd
import torch
import torch.nn as nn
from argument_generator import Argument_Generator
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MyDataset
from models.model import Model
from utils import cross_validation, threshold_function

# from torchsummary import summary


def train(args):

    # get state name
    t = time.localtime()
    run_time = time.strftime("%Y_%m_%d_%H_%M_%S", t)
    state_name = f"{args.model}_{args.optim}_{args.epochs}_{args.lr}_{args.batch_size}_{run_time}"

    # dataset
    full_set = MyDataset(args.train_path)
    num_classes = full_set.num_classes     # num_classes = 2537

    # use torch.random_split to create the validation dataset
    lengths = [int(round(len(full_set) * args.holdout_p)),
               int(round(len(full_set) * (1 - args.holdout_p)))]
    train_set, valid_set = random_split(full_set, lengths)

    # build hold out CV
    # train_set, valid_set = cross_validation(full_set, args.holdout_p)

    # choose training device
    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # load model
    model, parameters = Model().model_builder(
        args.model, num_classes, not args.non_pretrain, args.train_all)
    model = model.to(device)

    # set optimizer
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(
            parameters, lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(parameters, lr=args.lr)

    # set scheduler
    if args.scheduler is True:
        scheduler = ExponentialLR(optimizer, args.gamma)
    else:
        scheduler = None

    # set loss function
    criterion = nn.MSELoss(reduction='sum').to(device)
    # criterion = RMSELoss().to(device)
    # criterion = nn.BCELoss()

    # train
    loss_list = {'train': [], 'valid': []}
    accuracy_list = {'train': [], 'valid': []}
    dataloader = {'train': None, 'valid': None}
    best = 100

    # set dataloader
    dataloader['train'] = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )

    dataloader['valid'] = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # check output path exist
    save_path = os.path.join(args.output_path, state_name)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    args.output_path = save_path
    print(f"\nSaving output file and model parameters at {save_path}\n")

    # start to train
    for epoch in range(args.epochs):
        for phase in ['train', 'valid']:

            epoch_loss = 0.0
            correct = 0

            with tqdm(
                enumerate(dataloader[phase]),
                total=len(dataloader[phase]),
                desc=f'{epoch}/{args.epochs}, {phase}'
            ) as t, torch.set_grad_enabled(phase == 'train'):
                for _, data in t:
                    inputs, targets, labels = [d.to(device) for d in data]

                    # forward
                    outputs = model(inputs)
                    outputs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs.data, 1)

                    '''# get threshold values
                    if args.threshold != None:
                        outputs = threshold_function(outputs, args.threshold, device)'''

                    # count loss
                    loss = criterion(outputs.to(torch.float32),
                                     targets.to(torch.float32))

                    # backward
                    if phase == 'train':
                        optimizer.zero_grad()  # 清空上一輪算的 gradient
                        loss.backward()       # 計算 gradient
                        optimizer.step()      # 更新參數

                    epoch_loss += loss.item() * inputs.data.size(0)
                    correct += torch.sum(torch.flatten(preds)
                                         == torch.flatten(labels))

                    # lr scheduler every iteration
                    if args.iteration is True and args.scheduler is True and phase == 'train':
                        scheduler.step()

            epoch_loss /= len(dataloader[phase].dataset)
            loss_list[phase].append(epoch_loss)
            accuracy = float(correct) / len(dataloader[phase].dataset)
            accuracy_list[phase].append(accuracy)

            # learning rate scheduler
            if args.scheduler is True and args.iteration is False and phase == 'train':
                scheduler.step()

            if phase == 'valid' and epoch_loss < best:
                best = epoch_loss
                torch.save(model.state_dict(), os.path.join(
                    save_path, 'model_weights.pth'))

        print(
            f"Epoch {epoch+1}\tTrain Loss: {loss_list['train'][-1]:.4f}, Validation Loss: {loss_list['valid'][-1]:.4f}")
        print(
            f"Epoch {epoch+1}\tTrain Accuracy: {accuracy_list['train'][-1]:.4f}, Validation Accuracy: {accuracy_list['valid'][-1]:.4f}\n")

    # plot the loss curve for training and validation
    pd.DataFrame({
        "train-loss": loss_list['train'],
        "valid-loss": loss_list['valid']
    }).plot()
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, "Loss_curve.jpg"))

    # plot the accuracy curve for training and validation
    pd.DataFrame({
        "train-accuracy": accuracy_list['train'],
        "valid-accuracy": accuracy_list['valid']
    }).plot()
    plt.xlabel("Epoch"), plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_path, "Training_accuracy.jpg"))

    print(f"Best Validation Loss: {best}")
    print(f"\nSaving output files and model parameters at {save_path}")
    print("\nFinished Training\n")


if __name__ == '__main__':

    argument_generator = Argument_Generator()

    train(argument_generator)
