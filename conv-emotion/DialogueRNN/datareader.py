import os
import pickle
import re
import cv2
import operator
import numpy as np
from os import listdir
from scipy.io import wavfile
from moviepy.video.io.VideoFileClip import VideoFileClip
# from google.colab.patches import cv2_imshow


class DataReader(object):

    def __init__(self, dir_names, root_path='./raw_data/IEMOCAP_full_release', process_path=None):
        self.dir_names = dir_names
        self.root_path = root_path
        self.process_path = root_path if type(None) == type(process_path) else process_path
        self.label_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        # ignore label not in the mapping dict
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

    def read_label_file(self, label_root_path, f_name):
        label_file_path = label_root_path + '/' + f_name
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

    def read_transcription_file(self, transcription_root_path, f_name):
        transcription_file_path = transcription_root_path + '/' + f_name
        transcription_file = open(transcription_file_path, 'r')
        transcription_lines = transcription_file.readlines()

        count = 0
        u_ids = []
        u_speakers = []
        s_times = []
        e_times = []
        u_texts = []
        u_labels = []

        for line in transcription_lines:
            try:
                # process transcription file
                temp = re.split(' ', line, 2)
                u_id = temp[0]
                if self.ignore_dict[u_id] == 'xxx':
                    continue
                time = re.sub(r'[\[\]:]', '', temp[1]).split('-')
                s_time, e_time = float(time[0]), float(time[1])
                # print(transcription_file_path)
                # if 'Ses01M_impro04' in str(transcription_file_path):
                #     print(str(s_time) + ' - ' + str(e_time))
                text = re.sub(r'\n', '', temp[2])

                # TODO - implement text feature extraction
                # text_feature = text_extractor(text)

                speaker = u_id.split('_')[-1][0]

                u_ids.append(u_id)
                u_speakers.append(speaker)
                s_times.append(s_time)
                e_times.append(e_time)
                u_texts.append(text)
                count += 1
            except:
                # print(ses_path)
                # print(line)
                pass

        for u_id in u_ids:
            label = self.ignore_dict[u_id]
            if label != 'xxx':
                u_labels.append(self.label_map[label])

        return u_ids, u_speakers, s_times, e_times, u_texts, u_labels

    def read_audio_file(self, audio_root_path, f_name):
        u_audios = []
        wav_dir = re.sub('.txt', '', f_name)
        wav_file_names = listdir(audio_root_path + '/' + wav_dir)
        for wav_file in wav_file_names:
            # print(wav_file)
            if wav_file.startswith('.') or wav_file.endswith('.pk'):
                continue
            wav_file_path = audio_root_path + '/' + wav_dir + '/' + wav_file
            data = wavfile.read(wav_file_path)
            print(data)
            u_audios.append(data[1].astype('float64'))

        return u_audios

    def save_audio_pkl(self, utter_audio, pkl_name='utter_audio.pkl'):
        audio_pkl_path = self.process_path + '/read'
        if not os.path.isdir(audio_pkl_path):
            os.mkdir(audio_pkl_path)
        f = open(audio_pkl_path + '/' + pkl_name, 'wb')
        pickle.dump(utter_audio, f)
        f.close()

    def read_audio_pkl(self, pkl_name='utter_audio.pkl'):
        audio_pkl_path = self.process_path + '/read/' + pkl_name
        utter_audio = pickle.load(open(audio_pkl_path, 'rb'))
        new_dict = {}
        for k, v in utter_audio.items():
            # print(type(v))
            # print(type(v[0]))
            new_dict[k] = [x.astype('float32') for x in v]

        return new_dict

    def split_video_file(self, video_root_path, dir_name, f_name, u_id, s_time, e_time):
        avi_name = re.sub('.txt', '.avi', f_name)
        video_file_path = video_root_path + '/' + avi_name

        # check if the directory exist
        process_file_path = self.process_path + '/split/' + dir_name + '/' + re.sub('.txt', '', f_name)
        if not os.path.isdir(process_file_path):
            os.mkdir(process_file_path)

        target_file_path = process_file_path + '/' + u_id + '.mp4'

        with VideoFileClip(video_file_path) as video:
            s_time = s_time if float(s_time) >= float(video.start) else float(video.start)
            e_time = e_time if float(e_time) <= float(video.end) else float(video.end)
            new = video.subclip(s_time, e_time)
            new.write_videofile(target_file_path, audio_codec='aac')

    def read_video_file(self, dir_name, f_name, u_id):
        # only care about current speaker (not listener)
        prefix = u_id.split('_')[0]
        suffix = u_id.split('_')[-1]
        speaker_gender = 'M' if 'M' in suffix else 'F'
        left_speaker = 'M' if 'M' in prefix else 'F'

        split_video_path = self.process_path + '/split/' + dir_name + '/' + re.sub('.txt', '',
                                                                                   f_name) + '/' + u_id + '.mp4'
        # print(split_video_path)
        capture = cv2.VideoCapture(split_video_path)  # height * width * frames * channel
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

        video_frames = []
        # Read until video is completed
        while (capture.isOpened()):
            # Capture frame-by-frame
            ret, frame = capture.read()
            if ret == True:
                # Display the resulting frame
                if speaker_gender == left_speaker:
                    # print('left')
                    section = frame[int(height * (1 / 4)) + 10:int(height * (3 / 4)) - 6, 68:int(width / 2) - 68, :]
                else:
                    # print('right')
                    section = frame[int(height * (1 / 4)) + 10:int(height * (3 / 4)) - 6,
                              int(width / 2) + 68: int(width) - 68, :]

                video_frames.append(section)

                # cv2_imshow(section)
                # cv2.waitKey()
            else:
                break

        video_frames = np.moveaxis(np.stack(video_frames), 3, 1)
        video_frames = video_frames.astype('float32')
        capture.release()
        cv2.destroyAllWindows()

        # left = capture[:, :int(width/2), :, :]
        # right = capture[:, int(width / 2):, :, :]

        return video_frames

    def get_num_of_files(self, dir_name):
        label_root_path = self.root_path + '/' + dir_name + '/dialog/EmoEvaluation'
        file_names = listdir(label_root_path)
        return len(file_names)

    def get_data(self, start_idx=0):
        path = self.root_path

        utter_ids = {}
        utter_labels = {}
        utter_speakers = {}
        utter_texts = {}
        utter_audio = {}
        utter_s_times = {}
        utter_e_times = {}
        utter_speaker_frames = {}
        ram_limit = 3

        # iter sessions
        for dir_name in self.dir_names:
            # get label from DialogueRNN paper, so that the assigned labels can be consistent
            # pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
            # videoIDs, videoSpeakers, videoLabels, _, _, _, videoSentence, _, _ = pickle.load(open(pkl_path, 'rb'), encoding='latin1')

            # label file root path
            label_root_path = path + '/' + dir_name + '/dialog/EmoEvaluation'
            # transcription file root path
            transcription_root_path = path + '/' + dir_name + '/dialog/transcriptions'
            # wav file root path
            audio_root_path = path + '/' + dir_name + '/sentences/wav'
            # avi file root path
            video_root_path = path + '/' + dir_name + '/dialog/avi/DivX'

            file_names = listdir(label_root_path)

            for idx, f_name in enumerate(file_names):
                if idx < start_idx:
                    continue
                if idx >= start_idx + ram_limit:
                    break
                label_file_path = label_root_path + '/' + f_name
                if (not os.path.isfile(label_file_path)) or (f_name.startswith('.')):
                    continue

                # read label file
                self.read_label_file(label_root_path, f_name)

                # read transcription file
                u_ids, u_speakers, s_times, e_times, u_texts, u_labels = self.read_transcription_file(
                    transcription_root_path, f_name)

                key_word = re.sub('.txt', '', f_name)
                utter_ids[key_word] = u_ids
                utter_speakers[key_word] = u_speakers
                utter_s_times[key_word] = s_times
                utter_e_times[key_word] = e_times
                utter_texts[key_word] = u_texts
                utter_labels[key_word] = u_labels

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

                ###  read audio file
                # utter_audio[key_word] = self.read_audio_file(audio_root_path, f_name)

                ### read video file
                # split the video according to the start time and end time of utterance
                # for idx, u_id in enumerate(u_ids):
                #     count += 1
                #     s_time = utter_s_times[key_word][idx]
                #     e_time = utter_e_times[key_word][idx]
                #     self.split_video_file(video_root_path, dir_name, f_name, u_id, s_time, e_time)

                # read the split video
                print(f_name)
                temp = []

                for idx, u_id in enumerate(u_ids):
                    # get left and right portions of the video
                    res = self.read_video_file(dir_name, f_name, u_id)
                    # print(res)
                    temp.append(res)
                utter_speaker_frames[key_word] = temp

        return utter_ids, utter_labels, utter_speakers, utter_texts, utter_audio, utter_speaker_frames

