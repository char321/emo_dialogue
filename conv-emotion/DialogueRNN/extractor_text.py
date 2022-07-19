import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaTokenizer


class TextDataset(Dataset):
    def __init__(self, tokenizer, utters):
        self.tokenizer = tokenizer
        self.utters = utters

    def __len__(self):
        return len(self.utters)

    def __getitem__(self, index):
        text = self.utters[index]

        return text

    def collate_fn(self, data):
        encodings = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=256)
        return encodings


class IEMOCAPRobertaModel(RobertaPreTrainedModel):

    # Initialisation
    # config: pre-trained model config (model name)
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
    def __init__(self, cuda=False, config='roberta-base', batch_size=30, kernel_size=4, stride=4):
        self.cuda = cuda
        self.tokenizer = RobertaTokenizer.from_pretrained(config)

        model = IEMOCAPRobertaModel.from_pretrained(config)
        self.model = model.cuda() if self.cuda else model

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

        res = []
        self.extractor.eval()
        with torch.no_grad():
            loader = tqdm(data_loader)

            for data in loader:

                # max length of utterance in the IEMOCAP is 107 (split by space)
                encodings = self.tokenizer(data, return_tensors='pt', padding=True, max_length=128)
                device = torch.device("cuda:0" if self.use_gpu else "cpu")
                encodings = encodings.to(device)

                output = self.extractor(**encodings)

                text_vector = output.pooler_output  # CLS token

                if self.use_gpu:
                    res.append(text_vector.cpu())
                else:
                    res.append(text_vector)

        res = np.concatenate(res, axis=0)

        return res