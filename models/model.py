import sys
import torch.nn as nn
from torchsummary import summary
from torchvision.models import vgg19, resnet50, densenet161, googlenet, inception_v3

from .MyCNN import MyCNN


def VGG19(num_classes, pretrain=True, all=False):
    model = vgg19(pretrained=pretrain)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model


def VGG19_2(num_classes, pretrain=True, all=False):
    model = vgg19(pretrained=pretrain)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[3] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, num_classes)

    return model


def ResNet(num_classes, pretrain=True, all=False):
    model = resnet50(pretrained=pretrain)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # 修改全連線層的輸出
    model.fc = nn.Linear(2048, num_classes)

    return model


def Densenet(num_classes, pretrain=True, all=False):
    model = densenet161(pretrained=pretrain)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # 修改全連線層的輸出
    model.classifier = nn.Linear(2208, num_classes)

    return model


def GoogleNet(num_classes, pretrain=True, all=False):
    model = googlenet(pretrained=pretrain, aux_logits=False)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(1024, num_classes)

    return model


def inceptionv3(num_classes, pretrain=True, all=False):
    model = inception_v3(pretrained=pretrain, aux_logits=False)

    # 把參數凍結
    if all is False:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(2048, num_classes)

    return model


class Model():

    model_list = ['VGG19', 'VGG19_2', 'ResNet',
                  'MyCNN', 'Densenet', 'GoogleNet', 'inceptionv3']

    def get_model_list(self):
        return self.model_list

    def check_model_name(self, name):
        if name not in self.model_list:
            model_string = '\', \''.join(self.model_list)
            sys.exit(
                f"ModelNameError: '{name}' is not acceptable. The acceptable models are \'{model_string}\'.")

    def model_builder(self, model_name, num_classes, pretrain=True, train_all=False):

        # check if model name is acceptable
        self.check_model_name(model_name)

        if pretrain is False:
            train_all = True

        # load model
        model = globals()[model_name](num_classes, pretrain, train_all)

        # 取得要更新的參數
        parameters = []
        for _, param in model.named_parameters():
            if param.requires_grad == True:
                parameters.append(param)

        return model, parameters


if __name__ == '__main__':

    # model_list= ['VGG19', 'VGG19_2', 'ResNet', 'MyCNN', 'Densenet', 'GoogleNet', 'inceptionv3']
    model = Model().model_builder(Model().get_model_list()[3])
    summary(model, input_size=(3,224,224), batch_size=1, device="cpu")
    # print(model)
    pass