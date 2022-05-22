import torch
from torch.nn.utils.rnn import pad_sequence
from tan.structures import TLGBatch, TLGTestBatch


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        # feats, queries, wordlens, ious2d, moments, idxs = transposed_batch
        if len(transposed_batch) == 6:
            feats, queries, wordlens, ious2d, num_tubes, idxs = transposed_batch

            return TLGBatch(
                feats=torch.cat(feats, dim=0).float(),
                queries=queries,
                wordlens=wordlens,
                all_iou2d=torch.cat(ious2d, dim=0),
                num_tubes=torch.tensor(num_tubes).long()
                # moments=moments,
            ), idxs

        else:
            feats, queries, wordlens, num_tubes, idxs = transposed_batch

            return TLGTestBatch(
                feats=torch.cat(feats, dim=0).float(),
                queries=queries,
                wordlens=wordlens,
                num_tubes=torch.tensor(num_tubes).long()
                # moments=moments,
            ), idxs
