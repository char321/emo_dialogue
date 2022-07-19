import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification


class AudioFeatureExtractor(object):
    def __init__(self, cuda=False, config='wav2vec2-base', sampling_rate=16000):

        model_dict = {
            # Pre-trained
            'wav2vec2-base': 'facebook/wav2vec2-base',
            'wav2vec2-large': {'name': 'facebook/wav2vec2-large',
                               'revision': '85c73b1a7c1ee154fd7b06634ca7f42321db94db'},
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
        self.cuda = cuda
        self.model = Wav2Vec2Model.from_pretrained(model_dict[config])
        self.processor = Wav2Vec2Processor.from_pretrained(model_dict[config])
        self.model2 = Wav2Vec2ForSequenceClassification.from_pretrained(model_dict[config])
        # bundle = torchaudio.pipelines.WAV2VEC2_BASE
        # self.model3 = bundle.get_model()
        if self.cuda:
            self.model = self.model.cuda()
        self.sampling_rate = sampling_rate

    def get_audio_feature_test(self, waveforms):
        temp = []
        for waveform in waveforms:
            # print('---')
            # print(getsizeof(x))
            features, _ = self.model.extract_features(waveform)
            res = features[-1]
            res = torch.max(res, dim=1).values
            # print(torch.reshape(new_res, (768,)).shape)
            # res = torch.reshape(res, (768,))
            # print(getsizeof(res))
            temp.append(res)

        # feature: 12 * batch size * frames * feature dimension
        # print(features[-1].shape)
        # TODO use this for feature extraction

        return temp

    def get_batch_feature(self, batch):
        input = self.processor(batch, sampling_rate=self.sampling_rate, return_tensors='pt')  # batch size * max length

        # for v in input:
        #     print(len(v))
        if self.cuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input = input.to(device)

        with torch.no_grad():
            outputs = self.model(**input)
            audio_vector = outputs.extract_features
            test = outputs.last_hidden_state
        print('---')
        print(test.shape)
        print(audio_vector.shape)
        return audio_vector

    def get_batch_feature2(self, batch):
        input = self.processor(batch, sampling_rate=self.sampling_rate, return_tensors='pt',
                               padding=True)  # batch size * max length

        # for v in input:
        #     print(len(v))
        if self.cuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input = input.to(device)
        with torch.no_grad():
            outputs = self.model2(**input, output_hidden_states=True)
            test = outputs.hidden_states
        print('---')
        print(test.shape)

        return test

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