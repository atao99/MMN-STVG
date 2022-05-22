import torch
from torch import nn
import torch.nn.functional as F

def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d: B, D, N, N
    weight = F.conv2d(mask2d[None, None, :, :].float(),
                          mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight


def get_padded_mask_and_weight(mask, conv):
    masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(), stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]  #conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight


class ProposalConv(nn.Module):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, output_size, mask2d, dataset):
        super(ProposalConv, self).__init__()
        self.num_stack_layers = num_stack_layers
        self.dataset = dataset
        self.mask2d = mask2d[None, None,:,:]
        # Padding to ensure the dimension of the output map2d
        #mask_kernel = torch.ones(1, 1, k, k).to(mask2d.device)
        first_padding = (k - 1) * num_stack_layers // 2
        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size)])
        '''
        self.weights = [
            mask2weight(mask2d, mask_kernel, padding=first_padding)
        ]
        '''
        self.convs = nn.ModuleList(
            [nn.Conv2d(input_size, hidden_size, k, padding=first_padding)]
        )
        for _ in range(num_stack_layers - 1):
            #self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
            self.bn.append(nn.BatchNorm2d(hidden_size))
        self.conv1x1_iou = nn.Conv2d(hidden_size, output_size, 1)
        self.conv1x1_contrastive = nn.Conv2d(hidden_size, output_size, 1)
        '''
        print('self.num_stack_layers', self.num_stack_layers)
        print('conv', len(self.convs))
        print('weights', len(self.weights))
        print('bn', len(self.bn))
        '''

    def forward(self, x):
        padded_mask = self.mask2d
        for i in range(self.num_stack_layers):
            '''
            #print(self.weights[i])
            #print('layer',i, 'input, std=',x.std())
            x = self.convs[i](x)
            #print('layer', i, 'after convs, std=', x.std())
            x = self.bn[i](x)
            #print('layer', i, 'after bn, std=', x.std())
            x = x.relu()
            #print('layer', i, 'after relu, std=', x.std())
            '''
            x = self.bn[i](self.convs[i](x)).relu()
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.convs[i])
            x = x * masked_weight
            '''
            if self.dataset == "activitynet" or self.dataset == "charades":
                x = self.bn[i](self.convs[i](x)).relu() * self.weights[i]
            else:
                x = self.convs[i](x).relu() * self.weights[i]
            '''
        out1 = self.conv1x1_contrastive(x)
        out2 = self.conv1x1_iou(x)
        return out1, out2


def build_proposal_conv(cfg, mask2d):
    input_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
    hidden_size = cfg.MODEL.TAN.PREDICTOR.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TAN.PREDICTOR.KERNEL_SIZE
    num_stack_layers = cfg.MODEL.TAN.PREDICTOR.NUM_STACK_LAYERS
    output_size = cfg.MODEL.TAN.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME
    return ProposalConv(input_size, hidden_size, kernel_size, num_stack_layers, output_size, mask2d, dataset_name)