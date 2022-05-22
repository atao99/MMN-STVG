import datetime
import logging
import os
import time
import gc
import torch
import torch.distributed as dist

from tan.data import make_data_loader
from tan.utils.comm import get_world_size, synchronize
from tan.utils.metric_logger import MetricLogger
from tan.engine.inference import inference
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from ..utils.comm import is_main_process


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    loss = loss.item()
    return loss


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    param_dict,
    max_norm=10
):

    tensorboard_dir = 'tensorboardX/' + cfg.DATASETS.NAME + '/'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer1 = SummaryWriter(tensorboard_dir + 'video')
    writer2 = SummaryWriter(tensorboard_dir + 'sent')
    writer3 = SummaryWriter(tensorboard_dir + 'ranking')
    writer4 = SummaryWriter(tensorboard_dir + 'pairwise_sent')

    if cfg.DATASETS.METRIC_TENSORBOARD:
        synchronize()
        result_dict = inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=cfg.DATASETS.TEST,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
        )
        synchronize()
        model.train()
        if is_main_process():
            metric_writers = {k: SummaryWriter(tensorboard_dir + k) for k in result_dict}

    logger = logging.getLogger("tan.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    model.train()
    start_training_time = time.time()
    end = time.time()
    max_iteration = len(data_loader)
    writer_count = 0

    for epoch in range(arguments["epoch"], max_epoch + 1):
        rest_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch
        data_loader.batch_sampler.sampler.set_epoch(epoch)
        if cfg.MODEL.TAN.TEXT_ENCODER.NAME == "BERT":
            if epoch <= cfg.SOLVER.FREEZE_BERT:
                for param in param_dict['bert']:
                    param.requires_grad_(False)
            else:
                for param in param_dict['bert']:
                    param.requires_grad_(True)
            logger.info("Start epoch {}. base_lr={:.1e}, bert_lr={:.1e}, bert.requires_grad={}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], str(param_dict['bert'][0].requires_grad)))
        elif cfg.MODEL.TAN.TEXT_ENCODER.NAME == "LSTM":
            logger.info("Start epoch {}. base_lr={:.1e}".format(epoch, optimizer.param_groups[0]["lr"]))
        else:
            raise NotImplementedError
        if epoch <= cfg.SOLVER.ONLY_IOU:
            logger.info("Using all losses")
        else:
            logger.info("Using only bce loss")

        for iteration, (batches, idx) in enumerate(data_loader):
            writer_count += 1
            iteration += 1
            batches = batches.to(device)
            #targets = targets.to(device)
            optimizer.zero_grad()

            #loss_iou = model(batches, cur_epoch=epoch)

            contr_weight = cfg.MODEL.TAN.LOSS.CONTRASTIVE_WEIGHT
            loss_vid, loss_sent, loss_iou = model(batches, cur_epoch=epoch)
            #loss_iou = model(batches, cur_epoch=epoch)
            loss_vid, loss_sent = loss_vid * contr_weight, loss_sent * contr_weight
            meters.update(loss_vid=reduce_loss(loss_vid.detach()), loss_sent=reduce_loss(loss_sent.detach()), loss_iou=reduce_loss(loss_iou.detach()))
            #meters.update(loss_iou=loss_iou.detach())

            loss = 0
            if epoch <= cfg.SOLVER.ONLY_IOU:
                loss += loss_iou
                loss += loss_vid + loss_sent
            else:
                #loss_iou = model(batches, cur_epoch=epoch)
                loss += loss_iou #* 0.5
                loss += (loss_vid + loss_sent) * 0.01

            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            '''
            if is_main_process() and epoch <= cfg.SOLVER.ONLY_BCE:
                writer1.add_scalar('train_loss', loss_vid.detach().item(), writer_count)
                writer2.add_scalar('train_loss', loss_sent.detach().item(), writer_count)
                #writer3.add_scalar('train_statistics', loss_bce.detach().item(), writer_count)
                writer4.add_scalar('train_statistics', loss_pairwise_sent.detach().item(), writer_count)
                writer3.add_scalar('train_loss', loss_rank.detach().item(), writer_count)
                writer1.flush()
                writer2.flush()
                writer3.flush()
                writer4.flush()
            '''

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (max_iteration - iteration + rest_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            gc.collect()
        scheduler.step()

        if epoch % checkpoint_period == 0:
            checkpointer.save(f"{cfg.MODEL.TAN.FEAT2D.NAME}_model_{epoch}e", **arguments)

        if data_loader_val is not None and test_period > 0 and epoch % test_period == 0 and epoch >= cfg.SOLVER.SKIP_TEST:
            synchronize()
            torch.cuda.empty_cache()
            result_dict = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            synchronize()
            model.train()
            if cfg.DATASETS.METRIC_TENSORBOARD:
                if is_main_process():
                    for k, v in result_dict.items():
                        metric_writers[k].add_scalar('test_metrics', v, epoch)
                        metric_writers[k].flush()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
