# dogs-vs-cats

Author: Jeff Lin

---

## Summary

- [dogs-vs-cats](#dogs-vs-cats)
  - [Summary](#summary)
  - [Introduciotn](#introduciotn)
  - [Quick Run](#quick-run)
  - [Execution parameters](#execution-parameters)
  - [Experimental Results](#experimental-results)

---

## Introduciotn

This project is to practice the competition [Dogs vs. Cats Redux: Kernels Edition](#https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) on Kaggle. The competition is a Dogs vs. Cats classification problem.

Submissions are scored on the log loss:

<img src="https://i.imgur.com/D2CI0TM.png" />

where

- `n` is the number of images in the test set
- `\hat{y}_i` is the predicted probability of the image being a dog
- `y_i` is 1 if the image is a dog, 0 if cat
- `log` is the natural `base e` logarithm

Submission File should have a header and be in the following format:

```text
id,label
1,0.5
2,0.5
3,0.5
...

```

---

## Quick Run

Run the following command to run default training and evaluating.

```python
python main.py
```

---

## Execution parameters

See detail by `python main.py -h` command.

- `-h, --help`
  - show this help message and exit
- `--cuda CUDA`
  - set the model to run on which gpu (default: 0)
- `--holdout-p HOLDOUT_P`
  - set hold out CV probability (default: 0.8)
- `--num-workers NUM_WORKERS`
  - set the number of processes to run (default: 8)
- `--batch-size BATCH_SIZE`
  - set the batch size (default: 1)
- `--epochs EPOCHS`
  - set the epochs (default: 1)
- `--model MODEL_NAME`
  - set model name (default: 'VGG19')
  - The acceptable models are 'VGG19', 'VGG19_2', 'ResNet', 'MyCNN', 'Densenet', 'GoogleNet', 'inceptionv3'
- `--iteration`
  - set to decrease learning rate each iteration (default: False)
- `--train-all`
  - set to update all parameters of model (default: False)
- `--optim OPTIM`
  - set optimizer (default: SGD)
- `--lr LR`
  - set the learning rate (default: 1e-5)
- `--momentum MOMENTUM`
  - set momentum of SGD (default: 0.9)
- `--scheduler`
  - training with step or multi step scheduler (default: False)
- `--gamma GAMMA`
  - set decreate factor (default: 0.99985)
- `--threshold THRESHOLD`
  - the number thresholds the output answer
  - Float number >= 0 and <=1 (default: 0.99)
- `--output-path OUTPUT_PATH`
  - output file (csv, txt, pth) path (default: ./output)
- `--train-path TRAIN_PATH`
  - training dataset path (default: ./data/train/)
- `--test-path TEST_PATH`
  - evaluating dataset path (default: ./data/test1/)
- `--submit-csv SUBMIT_CSV`
  - submission CSV file (default: ./data/sample_submission.csv)
- `--kaggle Kaggle_Submission_Message`
  - the submission message to upload Kaggle.

---

## Experimental Results

The submission score of the best experimental result is **0.06038**. Click [here](https://drive.google.com/file/d/14A-P7tUS1nfKbAs1Z3SvrFCovybxlBE7/view?usp=sharing) to download the fine-trained model. The followings are the excution parameters of the best score.

|    Parameter    |  Value  |
| :-------------: | :-----: |
|  `--holdout-p`  |   0.8   |
| `--num-workers` |    8    |
| `--batch-size`  |    1    |
|   `--epochs`    |    1    |
|    `--model`    |  VGG19  |
|  `--iteration`  |  True   |
|  `--train-all`  |  True   |
|    `--optim`    |   SGD   |
|     `--lr`      |  1e-5   |
|  `--momentum`   |   0.9   |
|  `--scheduler`  |  True   |
|    `--gamma`    | 0.99985 |
|  `--threshold`  |  False  |
