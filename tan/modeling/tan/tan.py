import torch
from torch import nn
from torch.functional import F
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss #, build_iou_score_loss
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv


class TAN(nn.Module):
    def __init__(self, cfg):
        super(TAN, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.TAN.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.TAN.TEXT_ENCODER.NAME
        '''
        self.pairwise_sent_loss_weight = cfg.MODEL.TAN.LOSS.PAIRWISE_SENT_WEIGHT
        self.ranking_loss_weight = cfg.MODEL.TAN.LOSS.RANKING_WEIGHT
        # self.bce_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        # self.bce_loss_weight = cfg.MODEL.TAN.LOSS.BCE_WEIGHT
        '''
    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: Tensor of (batch_total_tubes, num_clips, num_clips)
            batches.feats: Tensor of (batch_total_tubes,num_pre_clips,C)

        # batch size for following tensors
        batch_total_tubes = num_tube_1 + num_tube_2 +...+ num_tube_b

        # size of some outputs:
        feats: Tensor of (batch_total_tubes, hidden_size, num_clips)
        vid_avg_feats: Tensor of (batch_total_tubes, joint_space_size)
        map2d: Tensor of (batch_total_tubes, hidden_size, num_clips, num_clips) after self.feat2d
        map2d: Tensor of (batch_total_tubes,joint_space_size,num_clips,num_clips) after self.proposal_conv
        sent_feat: len(sent_feat) = num_video, [(1, joint_space_size), (1, joint_space_size), ...], num_sent is 1 due to the setting of STVG dataset, for contrastive loss
        sen_feat_iou: len(sent_feat_iou) = num_video, [(1, joint_space_size), (1, joint_space_size), ...], for iou loss
        """

        # network forward pass
        feats, vid_avg_feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128,
        map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d, map2d_iou = self.proposal_conv(map2d)
        sent_feat, sent_feat_iou = self.text_encoder(batches.queries, batches.wordlens, vid_avg_feats)

        # compute predicted score
        contrastive_scores = []
        iou_scores = []
        _, T, _ = map2d[0].size()
        cumsum_num_tubes = torch.cumsum(batches.num_tubes, dim=0)
        cumsum_num_tubes = torch.cat([torch.tensor([0], dtype=torch.int64, device=cumsum_num_tubes.device), cumsum_num_tubes])

        for i, sf_iou in enumerate(sent_feat_iou):
            vid_feat_iou = map2d_iou[cumsum_num_tubes[i]:cumsum_num_tubes[i+1]]  # num_tube_i x C x T x T
            num_tube_i = vid_feat_iou.size(0)
            vid_feat_iou = vid_feat_iou.transpose(0, 1)  # C x num_tube_i x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou_norm = F.normalize(sf_iou, dim=1)  # 1 x C
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(num_tube_i, T, T)  # (num_tube_i, T, T)
            iou_scores.append((iou_score * 10).sigmoid() * self.feat2d.mask2d)  # (num_tube_i, T, T)

        # loss
        if self.training:
            ious2d = batches.all_iou2d
            loss_iou = self.iou_score_loss(torch.cat(iou_scores, dim=0), ious2d, cur_epoch)
            loss_vid, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d, cumsum_num_tubes)  # loss_pairwise_sent
            return loss_vid, loss_sent, loss_iou
            #return loss_iou
        else:
            for i, sf in enumerate(sent_feat):  # vid_avg_feats: B x C.  sent_feat_iou: [num_sent x C] (len=B)
                # contrastive part
                vid_feat = map2d[cumsum_num_tubes[i]:cumsum_num_tubes[i+1]].transpose(0, 1)  # C X num_tubes_i x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # 1 x C
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_tube_i x T x T
                contrastive_scores.append(contrastive_score)
            return contrastive_scores, iou_scores  #map2d_iou, sent_feat_iou,
            #return map2d_iou, sent_feat_iou, iou_scores
