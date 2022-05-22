import torch
from torch.functional import F
from tan.data.datasets.utils import box_iou
import torch.distributed as dist


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou, epoch):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    '''
    def scale(self, iou, min_iou=0.1, max_iou=0.95):
        return (iou - min_iou) / (max_iou - min_iou)

    def nonlinear_scale(self, iou, epoch):
        if epoch <= 10:
            coff = 2 + 0.1 * epoch
            return self.scale(torch.pow(iou, coff))
            #return torch.pow(iou, coff)
        else:
            return self.scale(torch.pow(iou, 5))
            #return torch.pow(iou, 3)
    '''

    def __call__(self, scores2d, ious2d, epoch):
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        iou1d = self.linear_scale(iou1d, epoch).clamp(0, 1)
        loss = self.bceloss(scores1d, iou1d).mean()
        return loss
        #return F.binary_cross_entropy_with_logits(scores2d.masked_select(self.mask2d), ious2d.masked_select(self.mask2d))


def build_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.TAN.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.TAN.LOSS.MAX_IOU
    return BceLoss(min_iou, max_iou, mask2d)


class ContrastiveLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.T_v = cfg.MODEL.TAN.LOSS.TAU_VIDEO
        self.T_s = cfg.MODEL.TAN.LOSS.TAU_SENT
        self.cri = torch.nn.CrossEntropyLoss()
        self.neg_iou = cfg.MODEL.TAN.LOSS.NEGATIVE_VIDEO_IOU
        self.top_k = cfg.MODEL.TAN.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL
        self.sent_removal_iou = cfg.MODEL.TAN.LOSS.SENT_REMOVAL_IOU
        self.margin = cfg.MODEL.TAN.LOSS.MARGIN
        self.eps = 1e-6
        self.use_gather = False
        self.use_arcFace = False
        self.arcFace_margin = 0.2
        self.dataset = cfg.DATASETS.NAME

    def __call__(self, feat2ds, sent_feats, iou2ds, cumsum_num_tubes):
        """
            feat2ds: Tensor of (batch_total_tubes, joint_space_size, num_clips, num_clips)
            iou2ds: Tensor of (batch_total_tubes, num_clips, num_clips)
        """
        # prepare tensors
        sent_feat_cat = torch.cat(sent_feats, 0)  # num_video x C, whole batch
        num_video = sent_feat_cat.size(0)
        sent_feat_cat_norm = F.normalize(sent_feat_cat, dim=1)  # num_video x C, whole batch

        B, C, _, _ = feat2ds.size()
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1)
        feat1ds_norm_cat = F.normalize(feat1ds, dim=1)  # B x C x num_sparse_selected_proposal
        num_proposal_per_tube = feat1ds_norm_cat.size(-1)
        feat1ds_list = []
        iou1d_list = []
        for i in range(cumsum_num_tubes.size(0)-1):
            feat1ds_list.append(feat1ds_norm_cat[cumsum_num_tubes[i]:cumsum_num_tubes[i+1]].permute(1, 0, 2))  # C X num_tube_i X num_sparse_selected_proposal
            iou1d_list.append(iou2ds[cumsum_num_tubes[i]:cumsum_num_tubes[i+1]].masked_select(self.mask2d).reshape(-1, num_proposal_per_tube))  # num_tube_i X num_sparse_selected_proposal
        feat1ds_norm = feat1ds_list

        assert len(feat1ds_norm) == len(sent_feats)  # = num_video

        sent_mask = torch.ones(num_video, num_video, device=feat2ds.device).bool()  # torch.ones: add all sentences into contrastive loss as default; torch.zeros: add none sentences as default

        '''
        all_num_sent = [0]
        curr_num_sent = 0
        for i in range(len(sent_feats)):
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)
        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            #print(i, gt_per_video.size(0), iou_mask.sum()/gt_per_video.size(0))
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).bool()  # add the sentence itself to the denominator in the loss
        '''

        if not self.use_arcFace:
            margin_mask = torch.diag(torch.ones(num_video, device=feat2ds.device)) * self.margin  # used for margin in contrastive loss

        vid_pos_list = []
        vid_neg_list = []
        sent_pos_list = []
        sent_neg_list = []
        vid_neg_gather_list = []

        for i, (sent_feat, iou2d) in enumerate(zip(sent_feats, iou2ds)):  # each video in the batch
            # select positive samples
            num_sent_this_batch = sent_feat.size(0)
            feat1d = feat1ds_norm[i]                                                                                # C x num_tube_i X num_sparse_selected_proposal
            sent_feat = F.normalize(sent_feat, dim=1)                                                               # 1 x C
            iou1d = iou1d_list[i]                                                                                   # num_tube_i X num_sparse_selected_proposal
            max_index = torch.argmax(iou1d)
            max_tube_index = max_index // iou1d.size(-1)
            selected_feat = feat1d.reshape(C, -1)[:, max_index].unsqueeze(0)                                         # 1 x C

            # positive video proposal with pos/neg sentence samples
            if self.use_arcFace:
                vid_pos = torch.acos(torch.bmm(selected_feat, sent_feat.unsqueeze(2)).reshape(-1).clamp(min=-1+self.eps, max=1-self.eps)) + self.arcFace_margin
                vid_pos = torch.cos(vid_pos)
            else:
                vid_pos = torch.mm(selected_feat, sent_feat.t()).squeeze(0) - self.margin                                      # 1 , bmm of (1 x C) and (C x 1)
            vid_neg = torch.mm(selected_feat, sent_feat_cat_norm.t()).reshape(-1, num_video)                        # 1 x num_video, mm of (1 x C) and (C x num_video)

            vid_pos_list.append(vid_pos)
            vid_neg_list.append(vid_neg)

            # positive sentence with pos/neg video proposals
            sent_pos_list.append(vid_pos.clone())
            sent_neg_same_video = torch.mm(sent_feat, feat1d[:, max_tube_index, :])                             # 1 x num_sparse_selected_proposal
            iou_neg_mask = (iou1d[max_tube_index] < self.neg_iou).float()                                       # only keep the low iou proposals as negative samples in the same video
            iou_max_mask = iou1d[max_tube_index].ge(iou1d.max() - self.eps).float()                        # mask of highest iou proposals (positive sample)

            sent_neg_same_video = (iou_neg_mask + iou_max_mask) * sent_neg_same_video                           # 1 x num_sparse_selected_proposal
            if self.use_arcFace:
                pos_index = iou_max_mask.nonzero().unbind(-1)
                sent_neg_same_video[pos_index] = torch.cos(torch.acos(sent_neg_same_video[pos_index].clamp(min=-1+self.eps, max=1-self.eps)) + self.arcFace_margin)
            else:
                sent_neg_same_video -= iou_max_mask * self.margin                                               # margin in contrastive loss

            feat1d_other_tube = torch.cat((feat1ds_norm_cat[:max_tube_index], feat1ds_norm_cat[max_tube_index+1:]))     # (num_total_tubes-1) x C x num_sparse_selected_proposal
            feat1d_other_tube = feat1d_other_tube.transpose(1, 0).reshape(C, -1)                                # C x ((num_total_tubes-1) * num_sparse_selected_proposal)

            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_tube)                                      # 1 x ((num_total_tubes-1) * num_sparse_selected_proposal)
            #print(vid_pos.clone().unsqueeze(1).size(), sent_neg_same_video.size(), sent_neg_other_video.size())
            sent_neg_all = [vid_pos.clone().unsqueeze(1), sent_neg_same_video, sent_neg_other_video]
            sent_neg_list.append(torch.cat(sent_neg_all, dim=1))  # 1 x (1 + num_same + num_other)

        vid_pos = torch.cat(vid_pos_list, dim=0) / self.T_v                                     # batch x 1
        vid_neg = torch.cat(vid_neg_list, dim=0)                                                # batch x num_video
        if self.use_arcFace:
            diag_mask = 1 - torch.diag(torch.ones(num_video, device=feat2ds.device))
            for i in range(vid_neg.size(0)):
                vid_pos_in_neg = torch.cos(torch.acos(torch.diag(vid_neg[i]).clamp(min=-1+self.eps, max=1-self.eps)) + self.arcFace_margin)
                vid_neg[i] = diag_mask * vid_neg + torch.diag(vid_pos_in_neg)
            vid_neg = vid_neg / self.T_v
        else:
            vid_neg = (vid_neg - margin_mask) / self.T_v                                            # this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)

        vid_neg_exp = torch.exp(vid_neg) * sent_mask
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=1, keepdim=False))).mean()
        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s
        sent_neg_exp = torch.exp(sent_neg)
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=1, keepdim=False) + self.eps)).mean()
        return loss_vid, loss_sent


def build_contrastive_loss(cfg, mask2d):
    return ContrastiveLoss(cfg, mask2d)
