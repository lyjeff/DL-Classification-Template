import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from argument_generator import Argument_Generator
from dataset import Eval_Dataset
from models.model import Model
from utils import label_decode, threshold_function


def eval(argument_generator):

    # get argument settings
    args = argument_generator.test_argument_setting()

    outputs_list = []
    confidences_list = []

    # dataset
    full_set = Eval_Dataset(args.test_path)

    # choose training device
    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # load model
    model, _ = Model().model_builder(args.model, num_classes=2537)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(
        args.output_path, 'model_weights.pth'), map_location=device))

    # set dataloader
    dataloader = DataLoader(
        full_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # start to evaluate
    model.eval()
    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Evaluation') as t, torch.set_grad_enabled(False):
        for _, inputs in t:
            inputs = inputs.to(device)

            # forward
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(outputs.data, 1)

            preds = preds.data.cpu().numpy().tolist()

            # get threshold values
            if args.threshold != None:
                outputs = threshold_function(outputs, args.threshold, device)

            outputs_list = outputs_list + preds
            confidences_list = confidences_list + confidences.data.cpu().numpy().tolist()

    # submit_csv['label'] = outputs_list
    # submit_csv.to_csv(os.path.join(
    #     args.output_path, 'answer.csv'), index=False)

    print(f"\nSaving output files in {args.output_path}")

    print("\nFinished Evaluating\n")


if __name__ == '__main__':

    argument_generator = Argument_Generator()

    eval(argument_generator)
