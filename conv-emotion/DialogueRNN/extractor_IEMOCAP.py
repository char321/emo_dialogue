import torch
import numpy as np
import pandas as pd
from torch.nn import MaxPool1d
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaTokenizer, RobertaConfig, \
    Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2Model
from os import listdir
import os
from os.path import isfile, join
from scipy.io import wavfile
import re
from dataloader import IEMOCAPDataset
import pickle
import collections
import operator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

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

class AudioFeatureExtractor(object):
    def __init__(self, config='wav2vec2-base-960h', feature_dim=100, sampling_rate=16000):
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
        # self.tokenizer = Wav2Vec2CTCTokenizer()
        # self.extractor = Wav2Vec2FeatureExtractor(feature_dim=feature_dim, sampling_rate=sampling_rate)
        # self.processor = Wav2Vec2Processor(self.extractor, self.tokenizer)
        self.model = Wav2Vec2Model.from_pretrained(model_dict[config])
        self.processor = Wav2Vec2Processor.from_pretrained(model_dict[config])
        self.sampling_rate = sampling_rate

    def get_audio_feature(self, audio):
        print('1')
        input = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors='pt', padding=True)
        print('2')
        with torch.no_grad():
            outputs = self.model(**input)

        print(outputs.extract_features)
        print('3')
        return outputs.extract_features


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


class DataReader():
    def __init__(self, dir_names, root_path='./raw_data/IEMOCAP_full_release'):
        self.dir_names = dir_names
        self.root_path = root_path
        self.label_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        self.ignore_dict = {}  # map 'id' to 'label'

    def assign_majority(self, e1, e2, e3, e4):
        temp_e1 = e1.split('\t')[1].replace(' ', '').split(';')
        temp_e2 = e2.split('\t')[1].replace(' ', '').split(';')
        temp_e3 = e3.split('\t')[1].replace(' ', '').split(';')
        temp_e4 = e4.split('\t')[1].replace(' ', '').split(';')

        # e = e1 + e2 + e3 + e4
        # e = temp_e1[:1] + temp_e2[:1] + temp_e3[:1] + temp_e4[:1]
        e = temp_e1 + temp_e2 + temp_e3 + temp_e4[:1]

        temp = {}
        for label in e:
            if label != '':
                try:
                    temp[label] += 1
                except:
                    temp[label] = 1

        sorted_temp = dict(sorted(temp.items(), key=operator.itemgetter(1), reverse=True))
        values = list(temp.values())
        max_num = values.count(max(values))

        remap = {'Excited': 'exc', 'Neutral': 'neu', 'Happiness': 'hap', 'Anger': 'ang', 'Frustration': 'fru'}
        label = list(sorted_temp.keys())[0]

        if label not in remap.keys():  # or sorted_temp[label] <= 2:
            return 'xxx'
        elif max_num > 1:
            return 'xxx'
            # return assign_majority_again(e1, e2, e3, e4)
        else:
            # return remap[label]
            if remap[label] == 'hap':
                # idk why code of DialogueRNN do this
                return remap[label]
            else:
                return 'xxx'

    def read_label_file(self, label_file_path):
        label_file = open(label_file_path, 'r')
        label_lines = label_file.readlines()

        # time_to_id = {}
        # time_to_label = {}
        # time_to_speaker = {}
        # time_to_sentence = {}

        ### generate filter mask - instance with no majority label should be ignored
        for line_idx, line in enumerate(label_lines):
            if not line.startswith('['):
                continue

            temp = line.split('\t')  # time; id; label; dim
            temp_id = temp[1]
            temp_label = temp[2]
            if temp_label not in self.label_map.keys():
                if temp_label == 'xxx':
                    # no majority - try to resign majority label
                    new_label = self.assign_majority(label_lines[line_idx + 1],
                                                     label_lines[line_idx + 2],
                                                     label_lines[line_idx + 3],
                                                     label_lines[line_idx + 4])

                    if new_label == 'xxx':
                        self.ignore_dict[temp_id] = 'xxx'
                    else:
                        self.ignore_dict[temp_id] = new_label
                else:
                    # other emotion
                    self.ignore_dict[temp_id] = 'xxx'
            else:
                self.ignore_dict[temp_id] = temp_label

    def read_transcription_file(self, transcription_file_path):
        transcription_file = open(transcription_file_path, 'r')
        transcription_lines = transcription_file.readlines()

        count = 0
        u_ids = []
        u_speakers = []
        s_times = []
        e_times = []
        u_texts = []

        for line in transcription_lines:
            try:
                # process transcription file
                temp = re.split(' ', line, 2)
                u_id = temp[0]
                if self.ignore_dict[u_id] == 'xxx':
                    continue
                time = re.sub(r'[\[\]:]', '', temp[1]).split('-')
                s_time, e_time = float(time[0]), float(time[1])
                text = re.sub(r'\n', '', temp[2])

                # TODO - implement text feature extraction
                # text_feature = text_extractor(text)

                speaker = u_id.split('_')[-1][0]

                u_ids.append(u_id)
                u_speakers.append(speaker)
                s_times.append(s_time)
                e_times.append(e_times)
                u_texts.append(text)
                count += 1
            except:
                # print(ses_path)
                # print(line)
                pass
        return u_ids, u_speakers, s_times, e_times, u_texts

    def get_data(self):
        path = self.root_path

        utter_ids = {}
        utter_labels = {}
        utter_speakers = {}
        utter_texts = {}
        utter_audio = {}

        # iter sessions
        for dir_name in dir_names:

            # ignore label not in the mapping dict
            label_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
            # label_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5, 'sur': 0}

            # get label from DialogueRNN paper, so that the assigned labels can be consistent
            # pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
            # videoIDs, videoSpeakers, videoLabels, _, _, _, videoSentence, _, _ = pickle.load(open(pkl_path, 'rb'), encoding='latin1')

            # label file root path
            label_root_path = path + '/' + dir_name + '/dialog/EmoEvaluation'
            file_names = listdir(label_root_path)

            # transcription file root path
            transcription_root_path = path + '/' + dir_name + '/dialog/transcriptions'

            # wav file root path
            wav_root_path = path + '/' + dir_name + '/sentences/wav'

            for f_name in file_names:

                label_file_path = label_root_path + '/' + f_name
                if (not os.path.isfile(label_file_path)) or (f_name.startswith('.')):
                    continue
                # read label file
                self.read_label_file(label_file_path)

                # read transcription file
                transcription_file_path = transcription_root_path + '/' + f_name
                u_ids, u_speakers, s_times, e_times, u_texts = self.read_transcription_file(transcription_file_path)

                u_labels = []
                for u_id in u_ids:
                    label = self.ignore_dict[u_id]
                    if label != 'xxx':
                        u_labels.append(label_map[label])

                key_word = re.sub('.txt', '', f_name)
                utter_ids[key_word] = u_ids
                utter_labels[key_word] = u_labels
                utter_speakers[key_word] = u_speakers
                utter_texts[key_word] = u_texts

                # test my obtains data vs data used in DialogueRNN
                # video_labels = videoLabels[key_word]
                # my_labels = utter_labels[key_word]
                # if len(my_labels) != len(video_labels):
                #     print(f_name)
                #     print(str(count) + '-' + str(len(my_labels)) + '-' + str(len(video_labels)))
                #     print(my_labels)
                #     print(video_labels)
                #     print(utter_ids[key_word])
                #     print(videoIDs[key_word])
                #     print(utter_speakers[key_word])
                #     print(videoSpeakers[key_word])
                #     print(utter_sentences[key_word])
                #     print(videoSentence[key_word])


                ###  read wav file
                u_audios = []
                wav_dir = re.sub('.txt', '', f_name)
                wav_file_names = listdir(wav_root_path + '/' + wav_dir)
                for wav_file in wav_file_names:
                    # print(wav_file)
                    if wav_file.startswith('.') or wav_file.endswith('.pk'):
                        continue
                    wav_file_path = wav_root_path + '/' + wav_dir + '/' + wav_file
                    # try:
                    data = wavfile.read(wav_file_path)
                    u_audios.append(data[1].astype('float64'))

                utter_audio[key_word] = u_audios

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
        return utter_ids, utter_labels, utter_speakers, utter_texts, utter_audio



if __name__ == '__main__':
    # define dims of features
    # d_t = 100  # text
    # d_v = 512  # visual
    # d_a = 100  # audio

    dir_names = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    dir_names = ['Session1']

    # utter_ids, utter_labels, utter_speakers, utter_texts, utter_audio = data_reader()
    data_reader = DataReader(dir_names)
    _, _, _, _, utter_audio = data_reader.get_data()

    # text feature
    # text_feature_extractor = TextFeatureExtractor()
    # utter_text_features = {}
    # for k, v in utter_texts.items():
    #     res = text_feature_extractor.get_text_feature(v)
    #     utter_text_features[k] = res

    # audio feature
    audio_feature_extractor = AudioFeatureExtractor()
    utter_audio_features = {}
    for k, v in utter_audio.items():
        print(k)
        res = audio_feature_extractor.get_audio_feature(v)
        utter_audio_features[k] = res


    # path = './IEMOCAP_features/text_feature.pkl'

    # with open(path, 'wb') as f:
    #     pickle.dump(utter_text_features, f)

    # temp = pickle.load(open(path, 'rb'), encoding='latin1')
    # print(len(temp.keys()))

    # for utters in list(utter_texts.values())[:1]:
    #     res = text_feature_extractor.get_text_feature(utters)
    #     print(len(res))
    #     print(res[0])


    # pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
    # videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(open(pkl_path, 'rb'),
    #                                                                              encoding='latin1')
    #
    # print(len(videoText.keys()))
    #
    # # print(list(temp.values())[0][0])
    # # print(list(videoText.values())[0][0])
    # for k, v in videoText.items():
    #     my_v = list(temp[k])
    #     og_v = list(v)
    #     og_av = list(videoAudio[k])
    #     og_vv = list(videoVisual[k])
    #     for idx, x in enumerate(my_v):
    #         print(sum(x))
    #         print(sum(og_v[idx]))
    #         print(sum(og_av[idx]))
    #         print(sum(og_vv[idx]))

    # print(videoSentence)
    # for k, v in videoText.items():
    #     print(len(v))
    #     print(len(v[0]))
    # print(videoSentence)