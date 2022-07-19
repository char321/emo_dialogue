import numpy as np

from extractor_IEMOCAP import *
from datareader import DataReader
import pickle

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    root_path = '/Users/chenyu/PycharmProjects/emo/conv-emotion/DialogueRNN/raw_data/IEMOCAP_full_release'
    process_path = '/Users/chenyu/PycharmProjects/emo/conv-emotion/DialogueRNN/raw_data/IEMOCAP_full_release'

    pkl_path = process_path + '/read/utter_audio_og.pkl'
    new_utter_audio = pickle.load(open(pkl_path, 'rb'))

    # Audio Feature Extraction
    utter_audio_feature = {}

    for k, v in new_utter_audio.items():
        audio_feature_extractor = AudioFeatureExtractor(False)
        utter_audio_feature[k] = audio_feature_extractor.get_batch_feature(v)

    # pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
    # videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, \
    # testVid = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
    #
    # for k, v in videoAudio.items():
    #     print(v)

    # dir_names = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    # # dir_names = ['Session1']
    # for dir_name in dir_names:
    #     data_reader = DataReader(dir_name)
    #     # utter_audio = data_reader.read_audio_pkl('utter_audio.pkl')
    #     utter_ids, utter_labels, utter_speakers, utter_texts, _, _ = data_reader.get_data()
    #
    #     for k, v in utter_labels.items():
    #         temp_v = videoLabels[k]
    #         if not temp_v == v:
    #             print(v)
    #             print(temp_v)
    #
    #         speaker = utter_speakers[k]
    #         temp_speaker = videoSpeakers[k]
    #
    #         if not temp_speaker == speaker:
    #             print(speaker)
    #             print(temp_speaker)
    #
    #         id = utter_ids[k]
    #         temp_id = videoIDs[k]
    #         if not temp_id == id:
    #             print(id)
    #             print(temp_id)

    # data_reader.save_audio_pkl(utter_audio, 'utter_audio.pkl')

    # print(list(utter_audio.keys()))
    # print(len(utter_audio.values()))
    # print(utter_audio['Ses01M_impro01'])
    # print(len(utter_audio['Ses01M_impro01']))

    # done - text feature
    # text_feature_extractor = TextFeatureExtractor()
    # utter_text_features = {}
    # for k, v in utter_texts.items():
    #     res = text_feature_extractor.get_text_feature(v)
    #     utter_text_features[k] = res

    # todo - audio feature
    # audio_feature_extractor = AudioFeatureExtractor(cuda)
    # utter_audio_features = {}
    # count = 0
    # for k, v in utter_audio.items():
    #     if count == 1:
    #         break
    #     # res = audio_feature_extractor.get_audio_feature(v)
    #     # print(res.shape)
    #     # utter_audio_features[k] = res
    #     print(k)
    #     print(v)
    #     print(len(v))
    #     print(len(v[0]))
    #     print(len(v[1]))
    #     count += 1

    # todo - video feature
    # video_feature_extractor = VideoFeatureExtractor()
    # frames = torch.Tensor(3, 3, 224, 224).normal_()  # random image
    # features = video_feature_extractor.get_video_features(frames)
    # print(features)
    # print(features.shape)

    # path = './IEMOCAP_features/text_feature.pkl'

    # with open(path, 'wb') as f:
    #     pickle.dump(utter_text_features, f)

    # temp = pickle.load(open(path, 'rb'), encoding='latin1')
    # print(len(temp.keys()))

    # for utters in list(utter_texts.values())[:1]:
    #     res = text_feature_extractor.get_text_feature(utters)
    #     print(len(res))
    #     print(res[0])

