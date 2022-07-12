import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.autograd import Variable
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaTokenizer, RobertaConfig, \
    Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2Model


class TextDataset(Dataset):
    def __init__(self, tokenizer, utters):
        self.tokenizer = tokenizer
        self.utters = utters
        # self.keys = list(utter_texts.keys())

    def __len__(self):
        # print('---len')
        return len(self.utters)

    def __getitem__(self, index):
        # print('---get_item')
        # c_id = self.keys[index]
        # text = self.utter_texts[c_id]
        # print(self.utters)
        text = self.utters[index]
        # print(text)
        # item = {'input_ids': text, 'attn_mask': None, 'output_path': None}
        # print(item)
        return text

    def collate_fn(self, data):
        # print('---collate_fn')
        # print(data)
        encodings = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=256)
        # print('---end_collate_fn')
        return encodings


class RobertaExtractor(RobertaPreTrainedModel):

  # Initialisation
  # config: pre-trained model config (model name)
  # dropout_rate: dropout rate
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class TextFeatureExtractor(object):
    def __init__(self, config='roberta-base', batch_size=30, kernel_size=4, stride=4):
        self.use_gpu = torch.cuda.is_available()
        self.tokenizer = RobertaTokenizer.from_pretrained(config)
        # model = RobertaPreTrainedModel.from_pretrained('roberta-base')
        # self.model = model.cuda() if self.use_gpu else model

        self.extractor = RobertaExtractor.from_pretrained(config)
        # self.extractor = RobertaPreTrainedModel.from_pretrained('roberta-base')
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.stride = stride

    def get_text_feature(self, utters):
        data_set = TextDataset(self.tokenizer, utters)
        data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            # collate_fn=data_set.collate_fn,
            drop_last=False,
            pin_memory=self.use_gpu
        )
        # print('mark 1')
        # print(data_loader)
        res = []
        self.extractor.eval()
        with torch.no_grad():
            loader = tqdm(data_loader)
            # print('mark 2')
            # print(loader)
            # print('mark 3')
            for data in loader:
                # print('mark 4')
                # print('===')
                print(data)
                # print(len(data))
                encodings = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True,
                                           max_length=256)
                # encodings.to(DEVICE)

                output = self.extractor(**encodings)
                # text_vector = output.last_hidden_state#[:, 0, :]
                text_vector = output.pooler_output
                pooling = torch.nn.AvgPool1d(self.kernel_size, self.stride)
                sampled_text_vector = pooling(text_vector)

                # print(torch.tensor(sampled_text_vector).shape)

                # print('mark 5')
                # res = self.extractor(data)
                # print('mark 5')
                res.append(sampled_text_vector)
        # print(len(res[0]))
        # print(len(res[1]))
        res = np.concatenate(res, axis=0)
        # print(len(res))
        return res


class AudioFeatureExtractor(object):
    def __init__(self, cuda=False, config='wav2vec2-base', feature_dim=100, sampling_rate=16000):

        model_dict = {
            # Pre-trained
            'wav2vec2-base': 'facebook/wav2vec2-base',
            'wav2vec2-large': {'name': 'facebook/wav2vec2-large', 'revision': '85c73b1a7c1ee154fd7b06634ca7f42321db94db'},
            # March 11, 2021 version: https://huggingface.co/facebook/wav2vec2-large/commit/85c73b1a7c1ee154fd7b06634ca7f42321db94db
            'wav2vec2-large-lv60': 'facebook/wav2vec2-large-lv60',
            'wav2vec2-large-xlsr-53': {'name': 'facebook/wav2vec2-large-xlsr-53',
                                       'revision': '8e86806e53a4df405405f5c854682c785ae271da'},
            # May 6, 2021 version: https://huggingface.co/facebook/wav2vec2-large-xlsr-53/commit/8e86806e53a4df405405f5c854682c785ae271da

            # Fine-tuned
            'wav2vec2-base-960h': 'facebook/wav2vec2-base-960h',
            'wav2vec2-large-960h': 'facebook/wav2vec2-large-960h',
            'wav2vec2-large-960h-lv60': 'facebook/wav2vec2-large-960h-lv60',
            'wav2vec2-large-960h-lv60-self': 'facebook/wav2vec2-large-960h-lv60-self',
            'wav2vec2-large-xlsr-53-english': 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
            'wav2vec2-large-xlsr-53-tamil': 'manandey/wav2vec2-large-xlsr-tamil'
        }
        model = Wav2Vec2Model.from_pretrained(model_dict[config])
        self.processor = Wav2Vec2Processor.from_pretrained(model_dict[config])
        self.cuda = cuda
        if self.cuda:
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # self.model = model.to(device)
            self.model = model.cuda()
        self.sampling_rate = sampling_rate

    def get_batch_feature(self, batch):
        input = self.processor(batch, sampling_rate=self.sampling_rate, return_tensors='pt', padding='max_length', max_length=60000)
        for v in input:
            print(len(v))
        if self.cuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input = input.to(device)
            # input = torch.tensor(input)
            # input = input.cuda()
        with torch.no_grad():
            outputs = self.model(**input)
        return outputs.extract_features

    def get_audio_feature(self, audio, max_batch_size=32):
        res = []
        temp = audio
        while True:
            if len(temp) > 32:
                batch = temp[:max_batch_size]

                res.append(self.get_batch_feature(batch))
                temp = temp[max_batch_size]
            else:
                batch = temp
                res.append(self.get_batch_feature(batch))
                break
        return res


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
