import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
from tabulate import tabulate

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        path = './IEMOCAP_features/text_feature.pkl'
        self.videoText = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        # reformat the data
        # text = torch.FloatTensor(self.videoText[vid])
        text = torch.FloatTensor(np.stack(self.videoText[vid], axis=0))  # (seq_len * 100)
        visual = torch.FloatTensor(np.stack(self.videoVisual[vid], axis=0))  # (seq_len * 512)
        audio = torch.FloatTensor(np.stack(self.videoAudio[vid], axis=0))  # (seq_len * 100)

        # [1, 0] - M; [0, 1] - F -> # (seq_len * 2)
        speakers = torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]])

        len_labels = torch.FloatTensor([1] * len(self.videoLabels[vid]))  # seq_len
        labels = torch.LongTensor(self.videoLabels[vid])  # seq_len

        return text, visual, audio, speakers, len_labels, labels, vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        df = pd.DataFrame(data)

        # reformat data's different features
        # text, visual, audio, speakers, len_labels, labels, vid
        return [pad_sequence(df[i]) if i < 4 else pad_sequence(df[i], True) if i < 6 else df[i].tolist() for i in df]


class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence,\
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'),encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, n_classes, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, _ = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return torch.LongTensor(self.InputSequence[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.ActLabels[conv])), \
                torch.LongTensor(self.ActLabels[conv]), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                self.InputMaxSequenceLength[conv], \
                conv

    def __len__(self):
        return self.len
    

class DailyDialoguePadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # stack all
        return torch.stack(batch, dim=0)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i==0 else \
                pad_sequence(dat[i]) if i == 1 else \
                pad_sequence(dat[i], True) if i < 5 else \
                dat[i].tolist() for i in dat]
