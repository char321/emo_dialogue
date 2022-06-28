import torch
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaTokenizer
from os import listdir
from os.path import isfile, join
from scipy.io.wavfile import read
import re


def text_extractor(text):
    # use pre-trained model
    # or training roberta on this dataset -> hidden layer provide feature

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # roberta = RobertaPreTrainedModel.from_pretrained('roberta-base')


    return text

def data_reader():
    path = './raw_data/IEMOCAP_full_release'

    dir_names = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    dir_names = ['Session1']

    # iter sessions
    for dir_name in dir_names:
        # read transcript file
        transcription_path = path + '/' + dir_name + '/dialog/transcriptions'
        wav_path = path + '/' + dir_name + '/sentences/wav'

        # iter utterances transcription file
        file_names = listdir(transcription_path)
        for f_name in file_names:
            ses_path = transcription_path + '/' + f_name
            file = open(ses_path, 'r')
            lines = file.readlines()

            descriptions = []
            s_times = []
            e_times = []
            text_features = []
            wav_features = []

            # iter lines in txt file
            for line in lines:

                try:
                    # process transcription file
                    temp = re.split(' ', line, 2)
                    des = temp[0]
                    time = re.sub(r'[\[\]:]', '', temp[1]).split('-')
                    s_time, e_time = float(time[0]), float(time[1])
                    text = temp[2]
                    text_feature = text_extractor(text)

                    descriptions.append(des)
                    s_times.append(s_time)
                    e_times.append(e_times)
                    text_features.append(text_feature)
                except:
                    print(ses_path)
                    print(line)

        # iter utterances wav file
        file_names = listdir(wav_path)
        for f_name in file_names:
            if f_name.startswith('.'):
                continue
            ses_path = wav_path + '/' + f_name
            ses_utter_names = listdir(ses_path)
            for ses_utter_name in ses_utter_names:
                if ses_utter_name.startswith('.'):
                    continue
                ses_utter_path = ses_path + '/' + ses_utter_name
                try:
                    wav_data = read(ses_utter_path)
                except:
                    print(ses_utter_path)




if __name__ == '__main__':
    # define dims of features
    d_t = 100  # text
    d_v = 512  # visual
    d_a = 100  # audio

    data_reader()
