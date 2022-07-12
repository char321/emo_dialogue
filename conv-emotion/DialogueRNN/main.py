from extractor_IEMOCAP import *
from datareader import DataReader

if __name__ == '__main__':
    # define dims of features
    # d_t = 100  # text
    # d_v = 512  # visual
    # d_a = 100  # audio

    # cuda = torch.cuda.is_available()
    # if cuda:
    #     print('Running on GPU')
    # else:
    #     print('Running on CPU')

    dir_names = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    # dir_names = ['Session1']
    data_reader = DataReader(dir_names)
    # utter_audio = data_reader.read_audio_pkl('utter_audio.pkl')
    _, _, _, _, _, utter_speaker_frames = data_reader.get_data()

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