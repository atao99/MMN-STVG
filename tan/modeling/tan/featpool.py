import torch
from torch import nn

class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride, joint_size):
        super(FeatAvgPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)
        self.fc1 = nn.Linear(input_size, joint_size)
        self.fc2 = nn.Linear(joint_size, joint_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, T
        return self.pool(self.conv(x).relu()), self.fc2(self.fc1(x.mean(-1)).relu())

def build_featpool(cfg):
    INPUT_SIZE = [1024, 1024, 4608]
    input_size = cfg.MODEL.TAN.FEATPOOL.INPUT_SIZE
    for dim in input_size:
        assert dim in INPUT_SIZE
        INPUT_SIZE.remove(dim)
    input_size = sum(input_size)
    hidden_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TAN.FEATPOOL.KERNEL_SIZE  # 4 for anet, 2 for tacos, 16 for charades
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.TAN.NUM_CLIPS
    joint_size = cfg.MODEL.TAN.JOINT_SPACE_SIZE
    return FeatAvgPool(input_size, hidden_size, kernel_size, stride, joint_size)
