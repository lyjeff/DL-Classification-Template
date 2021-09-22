import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DogDataset
from models.model import Model
from utils import argument_setting, threshold_function


def eval(args):

    submit_csv = pd.read_csv(args.submit_csv, header=0)
    outputs_list = []

    # dataset
    full_set = DogDataset(args.test_path, submit_csv)

    # choose training device
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # load model
    model = Model().model_builder(args.model)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_weights.pth')))

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
            outputs = outputs[:, 1]

            # get threshold values
            if args.threshold != None:
                outputs = threshold_function(outputs, args.threshold, device)

            outputs = outputs.data.cpu().numpy().tolist()
            outputs_list = outputs_list + outputs

    submit_csv['label'] = outputs_list
    submit_csv.to_csv(os.path.join(args.output_path, 'answer.csv'), index=False)

    print("\nFinished Evaluating\n")

if __name__ == '__main__':

    args = argument_setting()

    eval(args)
