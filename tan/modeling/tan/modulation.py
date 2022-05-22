import torch
from torch import nn
from torch.functional import F


class Modulation(nn.Module):
    def __init__(self, video_size, text_size, hidden_size, output_size, dataset):
        super(Modulation).__init__()

        #text
        self.layernorm = nn.LayerNorm(text_size)
        self.fc1 = nn.Linear(text_size, 2 * output_size)
        #video
        self.conv1x1 = nn.Conv2d(video_size, 2 * output_size, 1)
        #modulation
        self.vector = nn.Embedding(1, hidden_size)
        self.v_fc = nn.Linear(video_size, hidden_size)
        self.t_fc = nn.Linear(text_size, hidden_size)

    def forward(self, sent_feats, vid_feat_map2d):
        '''
        :param sent_feats: [N, C] (len=B)
        :param vid_feat_map2d: [B, C, T, T]
        :return: list of [num_sent, C], len=Batch_size
        '''
        for sent_feat in sent_feats:
            rho = F.softmax(torch.mm(self.vector, (self.v_fc(vid_feat_map2d) + self.s_fc(sent_feat))))

        return sent_feats


def build_modulation(cfg):
    joint_space_size = cfg.MODEL.TAN.JOINT_SPACE_SIZE
    video_size = cfg.MODEL.TAN.PREDICTOR.HIDDEN_SIZE
    query_size = 768 if cfg.MODEL.TAN.TEXT_ENCODER.NAME == 'BERT' else cfg.MODEL.TAN.TEXT_ENCODER.QUERY_HIDDEN_SIZE   # bert
    dataset_name = cfg.DATASETS.NAME
    hidden_size = 128
    return Modulation(video_size, query_size, hidden_size,joint_space_size, dataset_name)
