import os
from os.path import join, dirname
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn.utils.rnn import pad_sequence
from .utils import video2feats, moment_to_iou2d, glove_embedding, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer

class ActivityNetDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, root, feat_file, num_pre_clips, num_clips, pre_query_size, text_encoder_name):
        super(ActivityNetDataset, self).__init__()
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        logger = logging.getLogger("tan.trainer")
        logger.info("Preparing data, please wait...")

        for vid, anno in annos.items():
            duration = anno['duration']
            # Produce annotations
            moments = []
            all_iou2d = []
            queries = []
            sentences = []
            word_lens = []
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)
                    if text_encoder_name == 'LSTM':
                        query = glove_embedding(sentence)
                        queries.append(query)
                        word_lens.append(torch.tensor(query.size(0)))

            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)
            if text_encoder_name == 'LSTM':
                queries = pad_sequence(queries).transpose(0, 1)
                word_lens = torch.stack(word_lens)
            elif text_encoder_name == 'BERT':
                queries, word_lens = bert_embedding(sentences, tokenizer)  # padded query of N*word_len, tensor of size = N
            else:
                raise NotImplementedError("No such %s encoder!" % text_encoder_name)

            assert moments.size(0) == all_iou2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)

            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,
                    'iou2d': all_iou2d,
                    'sentence': sentences,
                    'query': queries,
                    'wordlen': word_lens,
                    'duration': duration,
                }
             )

        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="activitynet")

    def __getitem__(self, idx):
        #feat = self.feats[self.annos[idx]['vid']]
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="activitynet")
        return feat, self.annos[idx]['query'], self.annos[idx]['wordlen'], self.annos[idx]['iou2d'], self.annos[idx]['moment'], idx
    
    def __len__(self):
        return len(self.annos)
    
    def get_duration(self, idx):
        return self.annos[idx]['duration']
    
    def get_sentence(self, idx):
        return self.annos[idx]['sentence']
    
    def get_moment(self, idx):
        return self.annos[idx]['moment']
    
    def get_vid(self, idx):
        return self.annos[idx]['vid']


'''
f, (ax1) = plt.subplots(figsize=(50, 40), nrows=1)
                sns.heatmap(iou2d, linewidths=0.00, ax=ax1, vmax=1, vmin=0, cmap='rainbow')
                if not os.path.exists('./visual'):
                    os.mkdir('./visual')
                plt.savefig('./visual/iou_map.jpg')
                assert 0
'''