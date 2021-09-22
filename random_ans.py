import torch
import random
import pandas as pd
from utils import argument_setting, threshold_function

def random_generator(submit_csv, threshold):

    submit_csv = pd.read_csv(submit_csv, header=0)
    labels = torch.tensor([])

    for _ in range(submit_csv.shape[0]):
        labels = torch.cat((labels, torch.tensor([random.random()])), 0)

    if threshold != None:
        labels = threshold_function(labels, threshold)

    submit_csv['label'] = labels.data.cpu().tolist()
    submit_csv.to_csv('answer.csv', index=False)


if __name__ == '__main__':
    args = argument_setting()

    random_generator(args.submit_csv, args.threshold)