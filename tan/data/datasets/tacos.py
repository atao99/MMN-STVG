import os
from os.path import join, dirname
import json
import logging
import torch
from .utils import video2feats, moment_to_iou2d, glove_embedding, bert_embedding, get_vid_feat
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizer

class TACoSDataset(torch.utils.data.Dataset):

    def __init__(self, root, ann_file, feat_file, num_pre_clips, num_clips, pre_query_size, text_encoder_name):
        super(TACoSDataset, self).__init__()
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        with open(ann_file,'r') as f:
            annos = json.load(f)

        self.annos = []
        logger = logging.getLogger("tan.trainer")
        logger.info("Preparing data, please wait...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        for vid, anno in annos.items():
            duration = anno['num_frames']/anno['fps']  # duration of the video
            # Produce annotations
            moments = []
            all_iou2d = []
            queries = []
            sentences = []
            word_lens = []

            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0]/anno['fps'], 0), min(timestamp[1]/anno['fps'], duration)])
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
            #print("word_lens",word_lens)
            assert moments.size(0) == all_iou2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)
            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,  # N * 2
                    'iou2d': all_iou2d,  # N * 128*128
                    'sentence': sentences,   # list, len=N
                    'query': queries,  # padded query, N*word_len*C for LSTM and N*word_len for BERT
                    'wordlen': word_lens,  # size = N
                    'duration': duration
                    #'num_sentence': moments.size(0)  # int N
                }
            )

        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="tacos")

    def __getitem__(self, idx):
        # feat = self.feats[self.annos[idx]['vid']]
        feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="tacos")
        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        iou2d = self.annos[idx]['iou2d']
        moment = self.annos[idx]['moment']
        #N = moment.size(0)
        #rand_sample = torch.randperm(N)[:50]
        #return feat, query[rand_sample], wordlen[rand_sample], iou2d[rand_sample], moment[rand_sample], idx
        return feat, query, wordlen, iou2d, moment, idx

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
