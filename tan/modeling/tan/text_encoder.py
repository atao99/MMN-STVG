import torch
from torch import nn
from torch.functional import F
from transformers import DistilBertModel, AlbertModel  #, RobertaModel, BertModel


class DistilBert(nn.Module):
    def __init__(self, joint_space_size, dataset):
        super().__init__()
        use_albert=False
        if use_albert:
            self.bert =AlbertModel.from_pretrained('./albert-xxlarge-v1')
            self.fc_out1 = nn.Linear(4096, joint_space_size)
            self.fc_out2 = nn.Linear(4096, joint_space_size)
            self.layernorm = nn.LayerNorm(4096)
        else:
            self.bert = DistilBertModel.from_pretrained('./distilbert-base-uncased')  # RobertaModel.from_pretrained('roberta-base')
            self.fc_out1 = nn.Linear(768, joint_space_size)
            self.fc_out2 = nn.Linear(768, joint_space_size)
            self.layernorm = nn.LayerNorm(768)
        self.dataset = dataset
        self.aggregation = "avg"  # cls, attention, avg
        if self.aggregation == "attention":
            self.mha = nn.MultiheadAttention(joint_space_size, 4, dropout=0.1)
            self.fc1 = nn.Linear(768, joint_space_size)
            self.fc_out1 = nn.Linear(joint_space_size, joint_space_size)
            #self.fc_out2 = nn.Linear(joint_space_size, joint_space_size)

    def forward(self, queries, wordlens, vid_avg_feats):
        '''
        Average pooling over bert outputs among words to be sentence feature
        :param queries:
        :param wordlens:
        :param vid_avg_feat: B x C
        :return: list of [num_sent, C], len=Batch_size
        '''
        sent_feat = []
        sent_feat_iou = []
        for query, word_len, vid_avg_feat in zip(queries, wordlens, vid_avg_feats):  # each sample (several sentences) in a batch (of videos)
            N, word_length = query.size(0), query.size(1)
            attn_mask = torch.zeros(N, word_length, device=query.device)
            for i in range(N):
                attn_mask[i, :word_len[i]] = 1  # including [CLS] (first token) and [SEP] (last token)

            bert_encoding = self.bert(query, attention_mask=attn_mask)[0]  # [N, max_word_length, C]  .permute(2, 0, 1)
            if self.aggregation == "cls":
                query = bert_encoding[:, 0, :]  # [N, C], use [CLS] (first token) as the whole sentence feature
                query = self.layernorm(query)
                out_iou = self.fc_out1(query)
                out = self.fc_out2(query)
            elif self.aggregation == "avg":
                avg_mask = torch.zeros(N, word_length, device=query.device)
                for i in range(N):
                    #avg_mask[i, 1:word_len[i]-1] = 1   # excluding [CLS] (first token) and [SEP] (last token)
                    avg_mask[i, :word_len[i]] = 1       # including [CLS] (first token) and [SEP] (last token)
                #avg_mask = avg_mask / (word_len.unsqueeze(-1) - 2)
                avg_mask = avg_mask / (word_len.unsqueeze(-1))
                bert_encoding = bert_encoding.permute(2, 0, 1) * avg_mask  # use avg_pool as the whole sentence feature
                query = bert_encoding.sum(-1).t()  # [N, C]
                query = self.layernorm(query)
                out_iou = self.fc_out1(query)
                out = self.fc_out2(query)
            elif self.aggregation == "attention":
                query = self.layernorm(bert_encoding)
                query = self.fc1(query).relu().permute(1, 0, 2)  # [max_word_length, N, C]
                vid_avg_feat = vid_avg_feat.unsqueeze(0).unsqueeze(0).expand(-1, N, -1)  # 1 x N x C
                attn_mask = (1 - attn_mask).bool()  # attention ignores elements when mask is True
                #print('vid_avg_feat', vid_avg_feat.size(), "query", query.size(), "attn_mask", attn_mask.size())
                query = self.mha(vid_avg_feat, query, query, key_padding_mask=attn_mask)[0]  # 1 x N x C
                #print("query", query.size())
                out_iou = self.fc_out1(query.squeeze(0))
                # for contrastive branch
                query = bert_encoding[:, 0, :]  # [N, C], use [CLS] (first token) as the whole sentence feature
                query = self.layernorm(query)
                out = self.fc_out2(query)
            else:
                raise NotImplementedError
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou


class LSTM(nn.Module):
    def __init__(self, query_input_size, query_hidden_size, joint_space_size, bidirectional, num_layers, dataset):
        super(LSTM, self).__init__()
        self.layernorm = nn.LayerNorm(query_hidden_size)
        self.fc_out1 = nn.Linear(query_hidden_size, joint_space_size)
        self.fc_out2 = nn.Linear(query_hidden_size, joint_space_size)
        if bidirectional:
            query_hidden_size //= 2
        self.lstm = nn.LSTM(
            query_input_size, query_hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True
        )
        self.dataset = dataset

    def encode_query(self, queries, wordlens):
        self.lstm.flatten_parameters()
        sent_feat = []
        sent_feat_iou = []
        for query, word_len in zip(queries, wordlens):  # each sample (several sentences) in a batch (of videos)
            query = self.lstm(query)[0]  # [N, word_length, C]
            N, word_length = query.size(0), query.size(1)
            attn_mask = torch.zeros(N, word_length, device=query.device)
            for i in range(N):
                attn_mask[i, :word_len[i]] = 1
            attn_mask = attn_mask / (word_len.unsqueeze(-1))
            query = query.permute(2, 0, 1) * attn_mask  # [C, N, word_length]
            query = query.sum(-1).t()
            query = self.layernorm(query)
            out = self.fc_out1(query)
            out_iou = self.fc_out2(query)

            '''
            if self.dataset == "activitynet":
                query = self.dropout(F.relu(self.fc1(query)))
                query = self.dropout(F.relu(self.fc2(query))) + query
            else:
                query = self.bn(F.relu(self.fc1(query)))
                query = self.bn(F.relu(self.fc2(query))) + query
            query = self.fc3(query)
            '''
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou

    def forward(self, queries, wordlens):
        queries = self.encode_query(queries, wordlens)
        #map2d = self.conv(map2d)
        # F.normalize(queries * map2d)
        return queries


def build_text_encoder(cfg):
    joint_space_size = cfg.MODEL.TAN.JOINT_SPACE_SIZE
    query_input_size = cfg.INPUT.PRE_QUERY_SIZE
    query_hidden_size = cfg.MODEL.TAN.TEXT_ENCODER.QUERY_HIDDEN_SIZE
    bidirectional = cfg.MODEL.TAN.TEXT_ENCODER.LSTM.BIDIRECTIONAL
    num_layers = cfg.MODEL.TAN.TEXT_ENCODER.LSTM.NUM_LAYERS
    dataset_name = cfg.DATASETS.NAME
    if cfg.MODEL.TAN.TEXT_ENCODER.NAME == 'LSTM':
        return LSTM(query_input_size, query_hidden_size, joint_space_size, bidirectional, num_layers, dataset_name)
    elif cfg.MODEL.TAN.TEXT_ENCODER.NAME == 'BERT':
        return DistilBert(joint_space_size, dataset_name)
    else:
        raise NotImplementedError("No such text encoder as %s" % cfg.MODEL.TAN.TEXT_ENCODER.NAME)
