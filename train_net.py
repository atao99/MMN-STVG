import argparse
import os
import random
import torch
from torch import optim
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from tan.config import cfg
from tan.data import make_data_loader
from tan.engine.inference import inference
from tan.engine.trainer import do_train
from tan.modeling import build_model
from tan.utils.checkpoint import TanCheckpointer
from tan.utils.comm import synchronize, get_rank
from tan.utils.imports import import_file
from tan.utils.logger import setup_logger
from tan.utils.miscellaneous import mkdir, save_config

def train(cfg, local_rank, distributed):
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True
        )
    #optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.LR)
    learning_rate = cfg.SOLVER.LR * 1.0
    param_dict = {}

    if cfg.MODEL.TAN.TEXT_ENCODER.NAME == "BERT":
        bert_params = []
        base_params = []
        for name, param in model.named_parameters():
            if "bert" in name:
                bert_params.append(param)
            else:
                base_params.append(param)

        param_dict = {'bert': bert_params, 'base': base_params}

        if cfg.DATASETS.NAME == "tacos" or cfg.DATASETS.NAME == "charades":
            optimizer = optim.AdamW([{'params': base_params},
                                    {'params': bert_params, 'lr': learning_rate * 0.1}], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-6)
        elif cfg.DATASETS.NAME == "activitynet":
            optimizer = optim.AdamW([{'params': base_params},
                                     {'params': bert_params, 'lr': learning_rate * 0.1}], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-6)  # 1e-6
        elif cfg.DATASETS.NAME == "stvg":
            optimizer = optim.AdamW([{'params': base_params},
                                     {'params': bert_params, 'lr': learning_rate * 0.1}], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-6)  # 1e-6
        else:
            raise NotImplementedError('No such %s dataset!' % cfg.DATASETS.NAME)

    elif cfg.MODEL.TAN.TEXT_ENCODER.NAME == "LSTM":
        lstm_params = []
        base_params = []
        for name, param in model.named_parameters():
            if "lstm" in name:
                lstm_params.append(param)
            else:
                base_params.append(param)
        optimizer = optim.AdamW([{'params': base_params},
                                 {'params': lstm_params, 'lr': learning_rate * 1}], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
        param_dict = {'base': base_params, 'lstm': lstm_params}
    else:
        raise NotImplementedError

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = TanCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    arguments = {"epoch": 1}

    if cfg.SOLVER.RESUME:
        arguments = {"epoch": cfg.SOLVER.RESUME_EPOCH}
        if cfg.DATASETS.NAME == "activitynet":
            weight_path = './outputs/%s_activitynet_64x64_k9l4/%s_model_%de.pth' % (cfg.MODEL.TAN.FEAT2D.NAME, cfg.MODEL.TAN.FEAT2D.NAME, cfg.SOLVER.RESUME_EPOCH - 1)
        elif cfg.DATASETS.NAME == "tacos":
            weight_path = './outputs/%s_tacos_128x128_k5l8/%s_model_%de.pth' % (cfg.MODEL.TAN.FEAT2D.NAME, cfg.MODEL.TAN.FEAT2D.NAME, cfg.SOLVER.RESUME_EPOCH - 1)
        elif cfg.DATASETS.NAME == "charades":
            weight_path = './outputs/%s_charades_16x16_k5l8/%s_model_%de.pth' % (cfg.MODEL.TAN.FEAT2D.NAME, cfg.MODEL.TAN.FEAT2D.NAME, cfg.SOLVER.RESUME_EPOCH - 1)
        elif cfg.DATASETS.NAME == "stvg":
            weight_path = './outputs/%s_stvg_16x16_k5l8/%s_model_%de.pth' % (cfg.MODEL.TAN.FEAT2D.NAME, cfg.MODEL.TAN.FEAT2D.NAME, cfg.SOLVER.RESUME_EPOCH - 1)
        else:
            raise NotImplementedError('No checkpoints for such %s dataset' % cfg.DATASETS.NAME)
        weight_file = torch.load(weight_path, map_location=torch.device("cpu"))
        model.load_state_dict(weight_file.pop("model"))
        for _ in range(1, cfg.SOLVER.RESUME_EPOCH):
            scheduler.step()
        #optimizer.load_state_dict(weight_file.pop("optimizer"))
        #scheduler.load_state_dict(weight_file.pop('scheduler'))

    #arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed)

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
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
        param_dict
    )
    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="Tan")
    parser.add_argument(
        "--config-file",
        default="configs/2dtan_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    seed = 25285
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("tan", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    #mp.set_start_method('spawn')
    main()
