from argparse import ArgumentParser

from models.model import Model


class Argument_Generator():
    class __Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return self.start <= other <= self.end

    def __get_common_argument(self):

        parser = ArgumentParser()

        parser.add_argument('--cuda', type=int, default=0,
                            help='set the model to run on which gpu (default: 0)')

        # dataset argument
        parser.add_argument('--num-workers', type=int, default=8,
                            help='set the number of processes to run (default: 8)')

        # training argument
        parser.add_argument('--batch-size', type=int, default=32,
                            help='set the batch size (default: 32)')
        parser.add_argument(
            '--model', type=str,
            choices=Model().get_model_list(),
            metavar='MODEL_NAME',
            default='VGG19',
            help=f'set model name.\nThe acceptable models are {Model().get_model_list()} (default: "VGG19")'
        )

        # post-processing argument
        parser.add_argument(
            '--threshold',
            type=float,
            choices=[self.__Range(0.0, 1.0)],
            default=0.1,
            metavar='THRESHOLD',
            help='the number thresholds the output answer (Float number >= 0 and <=1) (default: 0.1)'
        )

        parser.add_argument('--output-path', type=str, default='./output/',
                            help='output file (csv, txt, pth) path (default: ./output)')

        # for the compatiable
        return parser

    def train_argument_setting(self):

        # get normal argument settings
        parser = self.__get_common_argument()

        # dataset path setting
        parser.add_argument('--train-path', type=str, default='./data_tmp/',
                            help='training dataset path (default: ./data_tmp/)')

        # training argument
        parser.add_argument('--epochs', type=int, default=50,
                            help='set the epochs (default: 50)')
        parser.add_argument('--non-pretrain', action="store_true", default=False,
                            help='Set to do not load pre-trained model (default: False)')
        parser.add_argument('--iteration', action="store_true", default=False,
                            help='set to decrease learning rate each iteration (default: False)')
        parser.add_argument('--train-all', action="store_true", default=False,
                            help='set to update all parameters of model (default: False)')

        # optimizer argument
        parser.add_argument('--optim', type=str, default='Adam',
                            help='set optimizer (default: Adam)')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='set the learning rate (default: 1e-3)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='set momentum of SGD (default: 0.9)')

        # scheduler argument
        parser.add_argument('--scheduler', action="store_true", default=False,
                            help='training with step or multi step scheduler (default: False)')
        parser.add_argument('--gamma', type=float, default=0.9,
                            help='set decreate factor (default: 0.9)')

        args = parser.parse_args()

        return args

    def test_argument_setting(self):

        # get normal argument settings
        parser = self.__get_common_argument()

        # set to save confidence scores to file
        parser.add_argument('--confidence', action="store_true", default=False,
                            help='set to save confidence scores to file (default: False)')

        # dataset path setting
        parser.add_argument('--test-path', type=str, default='./data_color/',
                            help='evaluating dataset path (default: ./data_color/)')
        parser.add_argument('--submit-csv', type=str, default='./test/public/Task2_Public_String_Coordinate.csv',
                            help='submission CSV file (default: ./test/public/Task2_Public_String_Coordinate.csv)')

        args = parser.parse_args()

        return args
