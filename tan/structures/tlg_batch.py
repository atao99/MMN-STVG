from dataclasses import dataclass
import torch

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    queries: list
    wordlens: list
    all_iou2d: list
    num_tubes: torch.tensor
    # moments: list

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.feats = self.feats.to(device)
        self.queries = [query.to(device) for query in self.queries]
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.all_iou2d = self.all_iou2d.to(device)
        self.num_tubes = self.num_tubes.to(device)
        # self.moments = [moment.to(device) for moment in self.moments]
        return self
    

