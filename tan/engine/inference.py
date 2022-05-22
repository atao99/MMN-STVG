import logging
import time
import os
from tqdm import tqdm

import torch
import json

from tan.data.datasets.evaluation import evaluate, predict
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, dataset_name, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for batch in data_loader:  # use tqdm(data_loader) for showing progress bar
        batches, idxs = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            contrastive_output, iou_output = model(batches.to(device))
            #_, _, iou_output = model(batches.to(device))
            #_, _, _, iou_output = model(batches.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            contrastive_output, iou_output = [o.to(cpu_device) for o in contrastive_output], [o.to(cpu_device) for o in iou_output]
            #iou_output = [o.to(cpu_device) for o in iou_output]
        results_dict.update(
            {video_id: {'contrastive': result1, 'iou': result2} for video_id, result1, result2 in zip(idxs, contrastive_output, iou_output)}
            #{video_id: { 'iou': result2} for video_id,  result2 in zip(idxs,  iou_output)}
        )
    '''
    
    elif dataset_name == "tacos" or dataset_name == "charades":
        for batch in tqdm(data_loader):
            batches, idxs = batch
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = model(batches.to(device))
                if timer:
                    if not device.type == 'cpu':
                        torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
            results_dict.update(
                {video_id: result for video_id, result in zip(idxs, output)}
            )
    else:
        raise NotImplementedError("No such %s dataset!" % dataset_name)
    '''
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    idxs = list(sorted(predictions.keys()))
    if len(idxs) != idxs[-1] + 1:
        logger = logging.getLogger("tan.inference")
        logger.warning(
            "Number of samples that were gathered from multiple processes is not "
            "a contiguous set. Some samples might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in idxs]
    return predictions

def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        device="cuda",
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("tan.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    inference_timer = Timer()
    predictions = compute_on_dataset(model, data_loader, device, cfg.DATASETS.NAME, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({:.03f} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    '''
    if not is_main_process():
        return
    '''
    return evaluate(cfg, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh)

def predict_test(
        cfg,
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        device="cuda",
):
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("tan.test")
    dataset = data_loader.dataset
    logger.info("Start testing on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    inference_timer = Timer()
    predictions = compute_on_dataset(model, data_loader, device, cfg.DATASETS.NAME, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({:.03f} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    '''
    if not is_main_process():
        return
    '''
    predict(dataset, predictions)
    return