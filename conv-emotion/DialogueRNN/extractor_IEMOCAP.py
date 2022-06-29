import torch
import numpy as np
import pandas as pd
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaTokenizer
from os import listdir
import os
from os.path import isfile, join
from scipy.io.wavfile import read
import re
from dataloader import IEMOCAPDataset
import pickle
import collections
import operator


def text_extractor(text):
    # use pre-trained model
    # or training roberta on this dataset -> hidden layer provide feature

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # roberta = RobertaPreTrainedModel.from_pretrained('roberta-base')


    return text

class TextFeatureExtractor(object):
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaPreTrainedModel.from_pretrained('roberta-base')
        self.model = model.cuda() if self.use_gpu else model


def assign_majority(e1, e2, e3, e4):
    e1 = e1.split('\t')[1].replace(' ', '').split(';')
    e2 = e2.split('\t')[1].replace(' ', '').split(';')
    e3 = e3.split('\t')[1].replace(' ', '').split(';')
    e4 = e4.split('\t')[1].replace(' ', '').split(';')

    e = e1 + e2 + e3 + e4
    temp = {}
    for label in e:
        if label != '':
            try:
                temp[label] += 1
            except:
                temp[label] = 1

    sorted_temp = dict(sorted(temp.items(), key=operator.itemgetter(1), reverse=True))
    values = list(temp.values())
    count = values.count(max(values))

    remap = {'Excited': 'exc', 'Neutral': 'neu', 'Happiness': 'hap', 'Anger': 'ang', 'Frustration': 'fru'}
    label = list(sorted_temp.keys())[0]

    if count > 1 or label not in remap.keys() or sorted_temp[label] <= 2:
        return 'xxx'
    else:
        return remap[label]


def data_reader():
    path = './raw_data/IEMOCAP_full_release'

    dir_names = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    dir_names = ['Session1']

    # iter sessions
    for dir_name in dir_names:

        # ignore label not in the mapping dict
        label_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        # label_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5, 'sur': 0}

        # get label from DialogueRNN paper, so that the assigned labels can be consistent
        pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
        train_set = IEMOCAPDataset(path=pkl_path)
        videoIDs, videoSpeakers, videoLabels, _, _, _, videoSentence, _, _ = pickle.load(open(pkl_path, 'rb'),
                                                                                         encoding='latin1')
        ignore_dict = {}
        # TODO - store label according order of transcription

        utter_ids = {}
        utter_labels = {}
        utter_speakers = {}
        utter_sentences = {}

        # iter label file
        label_root_path = path + '/' + dir_name + '/dialog/EmoEvaluation'
        file_names = listdir(label_root_path)
        for f_name in file_names:
            label_file_path = label_root_path + '/' + f_name
            if (not os.path.isfile(label_file_path)) or (f_name.startswith('.')):
                continue
            file = open(label_file_path, 'r')
            lines = file.readlines()

            time_to_id = {}
            time_to_label = {}
            time_to_speaker = {}
            time_to_sentence = {}
            print(lines)
            # list of labels for all utterances in each conversation
            for line_idx, line in enumerate(lines):
                if not line.startswith('['):
                    continue

                temp = line.split('\t')  # time; description; label; dim
                if temp[2] not in label_map.keys():
                    if temp[2] == 'xxx':
                        # try to resign majority label
                        e1 = lines[line_idx + 1]
                        e2 = lines[line_idx + 2]
                        e3 = lines[line_idx + 3]
                        e4 = lines[line_idx + 4]
                        new_label = assign_majority(e1, e2, e3, e4)
                        if new_label == 'xxx':
                            ignore_dict[temp[1]] = True
                        else:
                            ignore_dict[temp[1]] = False

                            time_stamp = float(re.sub('\[', '', temp[0].split(' ')[0]))
                            # id
                            u_id = temp[1]
                            time_to_id[time_stamp] = u_id
                            # label
                            time_to_label[time_stamp] = label_map[new_label]
                            # speaker
                            speaker = u_id.split('_')[2][0]
                            time_to_speaker[time_stamp] = speaker
                else:
                    ignore_dict[temp[1]] = False

                    time_stamp = float(re.sub('\[', '', temp[0].split(' ')[0]))

                    # id
                    u_id = temp[1]
                    time_to_id[time_stamp] = u_id

                    # label
                    time_to_label[time_stamp] = label_map[temp[2]]

                    # speaker
                    speaker = u_id.split('_')[2][0]
                    time_to_speaker[time_stamp] = speaker

            # sorted list according time stamp
            sorted_label = []
            sorted_speaker = []
            sorted_id = []
            print(time_to_label)
            time_to_label = collections.OrderedDict(sorted(time_to_label.items()))

            for k, v in time_to_label.items():
                if f_name == 'Ses01F_impro07.txt':
                    print(k)

                sorted_label.append(v)
                sorted_id.append(time_to_id[k])
                sorted_speaker.append(time_to_speaker[k])

            # print('---')
            # print(f_name)
            # print(len(sorted_label))
            key_word = re.sub('.txt', '', f_name)

            utter_ids[key_word] = sorted_id
            utter_labels[key_word] = sorted_label
            utter_speakers[key_word] = sorted_speaker

        print(utter_labels)
        print(ignore_dict)

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
            key_word = re.sub('.txt', '', f_name)

            video_labels = videoLabels[key_word]
            my_labels = utter_labels[key_word]

            count = 0
            utter_ids = []
            for line in lines:

                try:
                    # process transcription file
                    temp = re.split(' ', line, 2)
                    u_id = temp[0]
                    if ignore_dict[u_id] == True:
                        continue
                    time = re.sub(r'[\[\]:]', '', temp[1]).split('-')
                    s_time, e_time = float(time[0]), float(time[1])
                    text = temp[2]
                    text_feature = text_extractor(text)

                    utter_ids.append(u_id)
                    s_times.append(s_time)
                    e_times.append(e_times)
                    text_features.append(text_feature)
                    count += 1
                except:
                    # print(ses_path)
                    # print(line)
                    pass



            if count != len(video_labels):
                if f_name == 'Ses01F_impro07.txt':
                    print(f_name)
                    print(my_labels)
                    print(video_labels)
                    print(str(count) + '-' + str(len(my_labels)) + '-' + str(len(video_labels)))
                    print(utter_ids[key_word])
                    print(videoIDs[key_word])
                    print(utter_speakers[key_word])
                    print(videoSpeakers[key_word])
                    print(utter_sentences[key_word])
                    print(videoSentence[key_word])

        # iter utterances wav file

        # file_names = listdir(wav_path)
        # for f_name in file_names:
        #     if f_name.startswith('.'):
        #         continue
        #     ses_path = wav_path + '/' + f_name
        #     ses_utter_names = listdir(ses_path)
        #     for ses_utter_name in ses_utter_names:
        #         if ses_utter_name.startswith('.'):
        #             continue
        #         ses_utter_path = ses_path + '/' + ses_utter_name
        #         try:
        #             # TODO
        #             wav_data = read(ses_utter_path)
        #         except:
        #             print(ses_utter_path)

        # iter labels file
        # emo_map = {'angry': 'ang', 'happy': 'hap', 'sad': 'sad', 'neutral': 'neu', 'frustrated': 'fru', 'excited': 'exc', fearful, surprised, disgusted, other}
        # label_dict = {'neu': 1, 'e': 2}
        # emo_to_num = {'Neutral': 0, ''}




if __name__ == '__main__':
    # define dims of features
    d_t = 100  # text
    d_v = 512  # visual
    d_a = 100  # audio

    data_reader()
