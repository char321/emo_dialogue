import os
import re
import cv2
import operator
import numpy as np
from os import listdir
from moviepy.video.io.VideoFileClip import VideoFileClip

class DataReader(object):

    def __init__(self, dir_name=None, frame_sampling_rate=1 / 10, root_path='./raw_data/IEMOCAP_full_release',
                 process_path=None):
        self.dir_name = dir_name
        self.frame_sampling_rate = frame_sampling_rate
        self.root_path = root_path
        self.process_path = root_path if type(None) == type(process_path) else process_path
        self.label_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        # ignore label not in the mapping dict
        self.ignore_dict = {}  # map 'id' to 'label'

        # label file root path
        self.label_root_path = self.root_path + '/' + self.dir_name + '/dialog/EmoEvaluation'
        # transcription file root path
        self.transcription_root_path = self.root_path + '/' + self.dir_name + '/dialog/transcriptions'
        # wav file root path
        self.audio_root_path = self.root_path + '/' + self.dir_name + '/sentences/wav'
        # avi file root path
        self.video_root_path = self.root_path + '/' + self.dir_name + '/dialog/avi/DivX'

    def assign_majority(self, e1, e2, e3, e4):
        temp_e1 = e1.split('\t')[1].replace(' ', '').split(';')
        temp_e2 = e2.split('\t')[1].replace(' ', '').split(';')
        temp_e3 = e3.split('\t')[1].replace(' ', '').split(';')
        temp_e4 = e4.split('\t')[1].replace(' ', '').split(';')

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

    def read_label_file(self, f_name):
        label_file_path = self.label_root_path + '/' + f_name
        label_file = open(label_file_path, 'r')
        label_lines = label_file.readlines()

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

    def read_transcription_file(self, f_name):
        transcription_file_path = self.transcription_root_path + '/' + f_name
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
                text = re.sub(r'\n', '', temp[2])

                speaker = u_id.split('_')[-1][0]

                u_ids.append(u_id)
                u_speakers.append(speaker)
                s_times.append(s_time)
                e_times.append(e_time)
                u_texts.append(text)
                count += 1
            except:
                pass

        for u_id in u_ids:
            label = self.ignore_dict[u_id]
            if label != 'xxx':
                u_labels.append(self.label_map[label])

        return u_ids, u_speakers, s_times, e_times, u_texts, u_labels

    def read_audio_file(self, f_name):
        from scipy.io import wavfile

        u_audios = []
        wav_dir = re.sub('.txt', '', f_name)
        wav_file_names = listdir(self.audio_root_path + '/' + wav_dir)
        for wav_file in wav_file_names:
            if wav_file.startswith('.') or wav_file.endswith('.pk'):
                continue
            wav_file_path = self.audio_root_path + '/' + wav_dir + '/' + wav_file
            data = wavfile.read(wav_file_path)
            u_audios.append(data[1].astype('float32'))

        return u_audios

    def read_audio_file_as_waveform(self, f_name):
        import torchaudio

        u_audios = []
        wav_dir = re.sub('.txt', '', f_name)
        wav_file_names = listdir(self.audio_root_path + '/' + wav_dir)
        for wav_file in wav_file_names:
            # print(wav_file)
            if wav_file.startswith('.') or wav_file.endswith('.pk'):
                continue
            wav_file_path = self.audio_root_path + '/' + wav_dir + '/' + wav_file
            waveform, sample_rate = torchaudio.load(wav_file_path)
            # metadata = torchaudio.info(wav_file_path)

            u_audios.append(waveform)

        return u_audios

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

        # default fps is 30
        video_frames = []
        # Read until video is completed
        i = 0
        sampling_interval = int(1 / self.frame_sampling_rate)
        while (capture.isOpened()):
            # Capture frame-by-frame
            ret, frame = capture.read()
            if ret == False:
                break
            if i % sampling_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Display the resulting frame
                if speaker_gender == left_speaker:
                    # print('left')
                    section = frame[int(height * (1 / 4)) + 10:int(height * (3 / 4)) - 6, 68:int(width / 2) - 68]
                else:
                    # print('right')
                    section = frame[int(height * (1 / 4)) + 10:int(height * (3 / 4)) - 6,
                              int(width / 2) + 68: int(width) - 68]

                video_frames.append(section)
                # cv2_imshow(section)
                # cv2.waitKey()
            i += 1

        video_frames = np.stack(video_frames)
        video_frames = np.expand_dims(video_frames, axis=1)
        video_frames = video_frames.astype('float32')
        # print(video_frames.shape)

        capture.release()
        cv2.destroyAllWindows()

        # left = capture[:, :int(width/2), :, :]
        # right = capture[:, int(width / 2):, :, :]

        return video_frames

    def get_data(self):
        path = self.root_path

        utter_ids = {}
        utter_labels = {}
        utter_speakers = {}
        utter_texts = {}
        utter_audio = {}
        utter_s_times = {}
        utter_e_times = {}
        utter_speaker_frames = {}

        # get label from DialogueRNN paper, so that the assigned labels can be consistent
        # pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
        # videoIDs, videoSpeakers, videoLabels, _, _, _, videoSentence, _, _ = pickle.load(open(pkl_path, 'rb'), encoding='latin1')

        file_names = listdir(self.label_root_path)

        for f_name in file_names:
            label_file_path = self.label_root_path + '/' + f_name
            if (not os.path.isfile(label_file_path)) or (f_name.startswith('.')):
                continue

            print(f_name)
            # read label file
            self.read_label_file(f_name)

            # read transcription file
            u_ids, u_speakers, s_times, e_times, u_texts, u_labels = self.read_transcription_file(f_name)

            key_word = re.sub('.txt', '', f_name)
            utter_ids[key_word] = u_ids
            utter_speakers[key_word] = u_speakers
            utter_s_times[key_word] = s_times
            utter_e_times[key_word] = e_times
            utter_texts[key_word] = u_texts
            utter_labels[key_word] = u_labels

            ###  read audio file
            utter_audio[key_word] = self.read_audio_file(f_name)

            ### read video file
            # split the video according to the start time and end time of utterance
            # for idx, u_id in enumerate(u_ids):
            #     count += 1
            #     s_time = utter_s_times[key_word][idx]
            #     e_time = utter_e_times[key_word][idx]
            #     self.split_video_file(video_root_path, dir_name, f_name, u_id, s_time, e_time)

            # read the split video
            # temp = []
            #
            # for idx, u_id in enumerate(u_ids):
            #     # get left and right portions of the video
            #     res = self.read_video_file(self.dir_name, f_name, u_id)
            #     # print(res)
            #     temp.append(res)
            # utter_speaker_frames[key_word] = temp

        return utter_ids, utter_labels, utter_speakers, utter_texts, utter_audio, utter_speaker_frames

    def get_partial_data(self, start_idx=0):
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

        # get label from DialogueRNN paper, so that the assigned labels can be consistent
        # pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'
        # videoIDs, videoSpeakers, videoLabels, _, _, _, videoSentence, _, _ = pickle.load(open(pkl_path, 'rb'), encoding='latin1')

        # label file root path
        label_root_path = path + '/' + self.dir_name + '/dialog/EmoEvaluation'
        # transcription file root path
        transcription_root_path = path + '/' + self.dir_name + '/dialog/transcriptions'
        # wav file root path
        audio_root_path = path + '/' + self.dir_name + '/sentences/wav'
        # avi file root path
        video_root_path = path + '/' + self.dir_name + '/dialog/avi/DivX'

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

            for u_id in u_ids:
                # get left and right portions of the video
                res = self.read_video_file(self.dir_name, f_name, u_id)
                # print(res)
                temp.append(res)
            utter_speaker_frames[key_word] = temp

        return utter_ids, utter_labels, utter_speakers, utter_texts, utter_audio, utter_speaker_frames
