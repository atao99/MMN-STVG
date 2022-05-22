from terminaltables import AsciiTable
from tqdm import tqdm
import logging

import torch
from tan.data import datasets
from tan.data.datasets.utils import iou, score2d_to_moments_scores
from tan.data.datasets.iou3dt import iou3dt
from tan.utils.comm import is_main_process
import json

def nms(moments, scores, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True

    return moments[~suppressed]


def interval_evaluate(cfg, dataset, predictions, nms_thresh, recall_metrics=(1, 5)):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if not is_main_process():
        return
    if cfg.DATASETS.NAME == "tacos":
        iou_metrics = (0.1, 0.3, 0.5)
    elif cfg.DATASETS.NAME == "activitynet":
        iou_metrics = (0.3, 0.5, 0.7)
    elif cfg.DATASETS.NAME == "charades":
        iou_metrics = (0.5, 0.7)
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("tan.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))

    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    num_clips = predictions[0]['iou'].shape[-1]
    table = [['R@{},IoU@{:.01f}'.format(i, torch.round(j * 100) / 100) for i in recall_metrics for j in iou_metrics]]

    recall_x_iou = torch.zeros(10, num_recall_metrics, num_iou_metrics)

    num_instance = torch.zeros(10)
    for idx, result2d in tqdm(enumerate(predictions)):  # each video
        score2d = result2d['iou']
        duration = dataset.get_duration(idx)
        gt_moments = dataset.get_moment(idx)

        for gt_moment, pred_score2d in zip(gt_moments, score2d):  # each sentence
            dur_ratio = torch.floor((gt_moment[1]-gt_moment[0])/duration)
            num_instance[dur_ratio] += 1
            candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
            moments = nms(candidates, scores, nms_thresh)

            for i, r in enumerate(recall_metrics):
                mious = iou(moments[:r], gt_moment)
                bools = mious[:, None].expand(r, num_iou_metrics) > iou_metrics
                recall_x_iou[dur_ratio][i] += bools.any(dim=0)
    recall_x_iou /= num_instance
    for ratio in range(10):
        table.append(['{:.02f}'.format(recall_x_iou[ratio][i][j] * 100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics * num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)

    result_dict = {}
    return result_dict

def evaluate(cfg, dataset, predictions, nms_thresh, recall_metrics=(1,)):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    if not is_main_process():
        return
    if cfg.DATASETS.NAME == "tacos":
        iou_metrics = (0.1, 0.3, 0.5)
        two_evalutation = True
    elif cfg.DATASETS.NAME == "activitynet":
        iou_metrics = (0.3, 0.5, 0.7)
        two_evalutation = True
    elif cfg.DATASETS.NAME == "charades":
        iou_metrics = (0.5, 0.7)
        two_evalutation = True
    elif cfg.DATASETS.NAME == "stvg":
        iou_metrics = (0.3, 0.5)
        two_evalutation = False
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)

    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("tan.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    num_clips = predictions[0]['iou'].shape[-1]
    table = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]

    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
    num_instance = 0
    total_stiou = 0
    total_tiou = 0
    for idx, result2d in tqdm(enumerate(predictions)):   # each video
        num_instance += 1
        score2d = result2d['iou'] * result2d['contrastive']
        # score2d = result2d['iou']
        # score2d = result2d['iou']*(result2d['contrastive']/2 + 0.5)
        # score2d: Tensor of (num_tubes, num_clips, num_clips)
        # duration = dataset.get_duration(idx)
        # gt_moments = dataset.get_moment(idx)
        gttube = dataset.get_gttube(idx)
        # gttube: Tensor of (num_frames, 5)
        tubes = dataset.get_tubes(idx)
        # max_score = score2d.max()
        st = 0
        ed = -1
        while st >= ed:
            max_score_index = score2d.argmax()
            tube_id = max_score_index // (num_clips * num_clips)
            tube = tubes[tube_id]
            max_score_index -= tube_id * num_clips * num_clips
            place = torch.tensor((max_score_index // num_clips, max_score_index % num_clips), dtype=torch.long)
            score2d[tube_id][place[0]][place[1]] = -1
            place[1] += 1
            place = place.float() / num_clips
            tube_len = tube.size(0)
            cadidate = torch.round(place * tube_len).long()
            st = cadidate[0]
            ed = cadidate[1]

        if st < ed:
            tube_proposal = tube[st:ed]
            # st_frame = int(tube_proposal[0][0].item())
            # ed_frame = int(tube_proposal[-1][0].item())
            # tube_proposal_full = torch.empty(ed_frame-st_frame+1, 5)
            # tube_proposal_full[-1] = tube_proposal[-1]
            # for i in range(tube_proposal.size(0) - 1):
            #     index = torch.linspace(0, 1, 7).reshape(7, 1)
            #     a = tube_proposal[i]
            #     b = tube_proposal[i + 1]
            #     c = (b - a).unsqueeze(0)
            #     # subproposal = (torch.matmul(index, c) + a).round().long()
            #     subproposal = torch.matmul(index, c) + a
            #     tube_proposal_full[i*6:(i+1)*6+1,:] = subproposal
            stiou = iou3dt(tube_proposal.numpy(), gttube.numpy())
            tiou = iou3dt(tube_proposal.numpy(), gttube.numpy(), temporalonly=True)
            hit = stiou > iou_metrics
            recall_x_iou[0] += hit
            total_stiou += stiou
            total_tiou += tiou

        # tubes: [Tensor of (n1, 5), Tensor of (n2, 5), ...]

        # for gt_moment, pred_score2d in zip(gt_moments, score2d):  # each sentence
        #     num_instance += 1
        #     candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
        #     moments = nms(candidates, scores, nms_thresh)
        #
        #     for i, r in enumerate(recall_metrics):
        #         mious = iou(moments[:r], gt_moment)
        #         bools = mious[:, None].expand(r, num_iou_metrics) > iou_metrics
        #         recall_x_iou[i] += bools.any(dim=0)


        # for i, r in enumerate(recall_metrics):
        #     mious = iou3dt(tube_proposal, gttube)
        #     bools = mious[:, None].expand(r, num_iou_metrics) > iou_metrics
        #     recall_x_iou[i] += bools.any(dim=0)

    recall_x_iou /= num_instance
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    viou = total_stiou / num_instance
    tiou = total_tiou / num_instance
    logger.info("viou: {}, tiou: {}".format(viou, tiou))

    if two_evalutation:
        recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
        num_instance = 0
        for idx, result2d in tqdm(enumerate(predictions)):  # each video
            score2d = result2d['contrastive']
            duration = dataset.get_duration(idx)
            gt_moments = dataset.get_moment(idx)

            for gt_moment, pred_score2d in zip(gt_moments, score2d):  # each sentence
                num_instance += 1
                candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
                moments = nms(candidates, scores, nms_thresh)
                for i, r in enumerate(recall_metrics):
                    mious = iou(moments[:r], gt_moment)
                    bools = mious[:, None].expand(r, num_iou_metrics) > iou_metrics
                    recall_x_iou[i] += bools.any(dim=0)
        recall_x_iou /= num_instance
        table.append(['{:.02f}'.format(recall_x_iou[i][j] * 100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])

        recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
        num_instance = 0
        for idx, result2d in tqdm(enumerate(predictions)):  # each video
            score2d = torch.pow(result2d['contrastive'] * 0.5 + 0.5, cfg.TEST.CONTRASTIVE_SCORE_POW) * result2d['iou']
            duration = dataset.get_duration(idx)
            gt_moments = dataset.get_moment(idx)

            for gt_moment, pred_score2d in zip(gt_moments, score2d):  # each sentence
                num_instance += 1
                candidates, scores = score2d_to_moments_scores(pred_score2d, num_clips, duration)
                moments = nms(candidates, scores, nms_thresh)
                for i, r in enumerate(recall_metrics):
                    mious = iou(moments[:r], gt_moment)
                    bools = mious[:, None].expand(r, num_iou_metrics) > iou_metrics
                    recall_x_iou[i] += bools.any(dim=0)
        recall_x_iou /= num_instance
        table.append(['{:.02f}'.format(recall_x_iou[i][j] * 100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])

    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
    result_dict = {}
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], torch.round(iou_metrics[j]*100)/100)] = recall_x_iou[i][j]
    best_r1 = sum(recall_x_iou[0])/num_iou_metrics
    # best_r5 = sum(recall_x_iou[1])/num_iou_metrics
    result_dict['Best_R1'] = best_r1
    # result_dict['Best_R5'] = best_r5
    return result_dict

def predict(dataset, predictions):

    if not is_main_process():
        return

    num_clips = predictions[0]['iou'].shape[-1]

    results = {}
    for idx, result2d in tqdm(enumerate(predictions)):  # each video
        video_id = dataset.get_vid(idx)
        results[video_id] = {}
        score2d = result2d['iou'] * result2d['contrastive']
        # score2d: Tensor of (num_tubes, num_clips, num_clips)
        # duration = dataset.get_duration(idx)
        # gt_moments = dataset.get_moment(idx)
        # gttube: Tensor of (num_frames, 5)
        tubes = dataset.get_tubes(idx)
        # max_score = score2d.max()
        st = 0
        ed = -1
        max_score = 0
        while st >= ed:
            max_score_index = score2d.argmax()
            tube_id = max_score_index // (num_clips * num_clips)
            tube = tubes[tube_id]
            max_score_index -= tube_id * num_clips * num_clips
            place = torch.tensor((max_score_index // num_clips, max_score_index % num_clips), dtype=torch.long)
            # max_score = score2d[tube_id][place[0]][place[1]].item()
            score2d[tube_id][place[0]][place[1]] = -1
            place[1] += 1
            place = place.float() / num_clips
            tube_len = tube.size(0)
            cadidate = torch.round(place * tube_len).long()
            st = cadidate[0]
            ed = cadidate[1]

        if st < ed:
            tube_proposal = tube[st:ed]
            # results[video_id]['score'] = max_score
            results[video_id]['st_frame'] = int(tube_proposal[0][0].item())
            results[video_id]['ed_frame'] = int(tube_proposal[-1][0].item())
            bbox = {}
            for i in range(tube_proposal.size(0)):
                bbox[str(int(tube_proposal[i][0].item()))] = tube_proposal[i][1:5].numpy().tolist()
            results[video_id]['bbox'] = bbox
            print(video_id, 'done', 'len:',results[video_id]['ed_frame']-results[video_id]['st_frame']+1)
        else:
            print(video_id, 'no prediction')

    with open('./results.json','w') as f:
        json.dump(results, f)
