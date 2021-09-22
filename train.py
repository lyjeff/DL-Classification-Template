import os
import time
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR
# from torchsummary import summary

from matplotlib import pyplot as plt

from dataset import CatDataset
from models.model import Model
from utils import argument_setting, threshold_function, cross_validation


def train(args):

    # get state name
    t = time.localtime()
    run_time = time.strftime("%Y_%m_%d_%H_%M_%S", t)
    state_name = f"{args.model}_{args.optim}_{args.epochs}_{args.lr}_{args.batch_size}_{run_time}"

    # dataset
    full_set = CatDataset(args.train_path)

    # use torch.random_split to create the validation dataset
    lengths = [int(round(len(full_set) * args.holdout_p)), int(round(len(full_set) * (1 - args.holdout_p)))]
    train_set, valid_set = random_split(full_set, lengths)

    # build hold out CV
    # train_set, valid_set = cross_validation(full_set, args.holdout_p)

    # choose training device
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # load model
    model = Model().model_builder(args.model, args.train_all)
    model = model.to(device)

    # 取得要更新的參數
    parameters = []
    for _,param in model.named_parameters():
        if param.requires_grad == True:
            parameters.append(param)

    # set optimizer
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum)

    # set scheduler
    if args.scheduler is True:
        scheduler = ExponentialLR(optimizer, args.gamma)
    else:
        scheduler = None

    # set loss function
    criterion = nn.BCELoss()

    # train
    loss_list = {'train':[], 'valid':[]}
    accuracy_list = {'train':[], 'valid':[]}
    dataloader={'train':None, 'valid':None}
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
    print(f"Saving output file and model parameters at {save_path}")

    # start to train
    for epoch in range(args.epochs):
        for phase in ['train', 'valid']:

            epoch_loss = 0.0
            correct = 0

            with tqdm(
                enumerate(dataloader[phase]),
                total=len(dataloader[phase]),
                desc=f'{epoch}/{args.epochs}, {phase}'
            ) as t, torch.set_grad_enabled(phase=='train'):
                for _, data in t:
                    inputs, targets = data[0].to(device), data[1].to(device)

                    # forward
                    outputs = model(inputs)
                    outputs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs.data, 1)
                    targets = targets.to(torch.float32)
                    outputs = outputs[:, 1]

                    # get threshold values
                    if args.threshold != None:
                        outputs = threshold_function(outputs, args.threshold, device)

                    # count loss
                    loss = criterion(outputs, targets)

                    # backward
                    if phase == 'train':
                        optimizer.zero_grad() # 清空上一輪算的 gradient
                        loss.backward()       # 計算 gradient
                        optimizer.step()      # 更新參數

                    epoch_loss += loss.item() * inputs.data.size(0)
                    correct += torch.sum(preds == targets.data)

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
                torch.save(model.state_dict(), os.path.join(save_path, 'model_weights.pth'))

        print(f"Epoch {epoch}\tTrain Loss: {loss_list['train'][-1]:.4f}, Validation Loss: {loss_list['valid'][-1]:.4f}")
        print(f"Epoch {epoch}\tTrain Accuracy: {accuracy_list['train'][-1]:.4f}, Validation Accuracy: {accuracy_list['valid'][-1]:.4f}\n")

    # plot the loss curve for training and validation
    pd.DataFrame({
        "train-loss": loss_list['train'],
        "valid-loss": loss_list['valid']
    }).plot()
    plt.xlabel("Epoch"),plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, "Loss_curve.jpg"))

    # plot the accuracy curve for training and validation
    pd.DataFrame({
        "train-accuracy": accuracy_list['train'],
        "valid-accuracy": accuracy_list['valid']
    }).plot()
    plt.xlabel("Epoch"),plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_path, "Training_accuracy.jpg"))

    print(f"Best Validation Loss: {best}")
    print("\nFinished Training\n")


if __name__ == '__main__':

    args = argument_setting()
    train(args)
