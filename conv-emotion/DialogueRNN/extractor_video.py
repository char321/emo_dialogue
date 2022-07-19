import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


class VideoFeatureExtractor(object):
    def __init__(self, cuda=False):
        densenet = models.densenet121(pretrained=True)
        num_ftrs = densenet.classifier.in_features
        for p in densenet.parameters():
            p.requires_grad = False
        densenet.classifier = nn.Flatten()
        self.cuda = cuda

        if cuda:
            self.densenet = densenet.cuda()
        else:
            self.densenet = densenet

        # resnet152 = models.densenet121(pretrained=True)
        # modules = list(resnet152.children())[:-1]  # remove the last layer
        # resnet152 = nn.Sequential(*modules)
        # for p in resnet152.parameters():
        #     p.requires_grad = False
        #
        # img = torch.Tensor(1, 3, 224, 224).normal_()  # random image
        # img_var = Variable(img)  # assign it to a variable
        # features_var = resnet152(img_var)  # get the output from the last hidden layer of the pretrained resnet
        # features = features_var.data  # get the tensor out of the variable
        #
        # print(features)
        # print(features.shape)

    def get_video_features(self, video_frames, pooling_type='mean'):
        img_var = Variable(torch.tensor(video_frames))
        if self.cuda:
            img_var = img_var.cuda()

        features_var = self.densenet(img_var)  # get the output from the last hidden layer of the pretrained resnet
        features = features_var.data  # get the tensor out of the variable

        if pooling_type == 'mean':
            res = torch.mean(features, 0, keepdims=True)
        if pooling_type == 'max':
            res = torch.max(features, 0, keepdims=True)
        return res
