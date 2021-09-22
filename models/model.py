import sys
import torch.nn as nn
from torchsummary import summary
from torchvision.models import vgg19, resnet50, densenet161, googlenet, inception_v3

from .MyCNN import MyCNN

def VGG19(all=False):
    model = vgg19(pretrained=True)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[6] = nn.Linear(4096, 2)

    return model


def VGG19_2(all=False):
    model = vgg19(pretrained=True)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[3] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, 2)

    return model


def ResNet(all=False):
    model = resnet50(pretrained=True)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # 修改全連線層的輸出
    model.fc = nn.Linear(2048, 2)

    return model


def Densenet(all=False):
    model = densenet161()

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # 修改全連線層的輸出
    model.classifier = nn.Linear(2208, 2)

    return model


def GoogleNet(all=False):
    model = googlenet(pretrained=True)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(1024, 2)

    return model


def inceptionv3(all=False):
    model = inception_v3(pretrained=True, aux_logits=False)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(2048, 2)

    return model


class Model():

    model_list = ['VGG19', 'VGG19_2', 'ResNet', 'MyCNN', 'Densenet', 'GoogleNet', 'inceptionv3']

    def get_model_list(self):
        return self.model_list

    def check_model_name(self, name):
        if name not in self.model_list:
            model_string = '\', \''.join(self.model_list)
            sys.exit(f"ModelNameError: '{name}' is not acceptable. The acceptable models are \'{model_string}\'.")

    def model_builder(self, model_name, train_all=False):

        # check if model name is acceptable
        self.check_model_name(model_name)

        # load model
        model = globals()[model_name](train_all)

        return model


if __name__ == '__main__':

    # model_list= ['VGG19', 'VGG19_2', 'ResNet', 'MyCNN', 'Densenet', 'GoogleNet', 'inceptionv3']
    model = Model().model_builder(Model().get_model_list()[3])
    summary(model, input_size=(3,224,224), batch_size=1, device="cpu")
    # print(model)
    pass