"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "tacos_train":{
            "video_dir": "/data0/wzz/data/TACoS/videos",
            "ann_file": "/data0/wzz/data/TACoS/train.json",
            "feat_file": "/data0/wzz/data/TACoS/tall_c3d_features.hdf5",
        },
        "tacos_val":{
            "video_dir": "/data0/wzz/data/TACoS/videos",
            "ann_file": "/data0/wzz/data/TACoS/val.json",
            "feat_file": "/data0/wzz/data/TACoS/tall_c3d_features.hdf5",
        },
        "tacos_test":{
            "video_dir": "/data0/wzz/data/TACoS/videos",
            "ann_file": "/data0/wzz/data/TACoS/test.json",
            "feat_file": "/data0/wzz/data/TACoS/tall_c3d_features.hdf5",
        },
        "activitynet_train":{
            "video_dir": "/data0/wzz/data/ActivityNet/videos",
            "ann_file": "/data0/wzz/data/ActivityNet/train.json",
            "feat_file": "/data0/wzz/data/ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "activitynet_val":{
            "video_dir": "/data0/wzz/data/ActivityNet/videos",
            "ann_file": "/data0/wzz/data/ActivityNet/val.json",
            "feat_file": "/data0/wzz/data/ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "activitynet_test":{
            "video_dir": "/data0/wzz/data/ActivityNet/videos",
            "ann_file": "/data0/wzz/data/ActivityNet/test.json",
            "feat_file": "/data0/wzz/data/ActivityNet/sub_activitynet_v1-3.c3d.hdf5",
        },
        "charades_train": {
            "video_dir": "/data0/wzz/data/Charades_STA/videos",
            "ann_file": "/data0/wzz/data/Charades_STA/charades_train.json",
            "feat_file": "/data0/wzz/data/Charades_STA/vgg_rgb_features.hdf5",
        },
        "charades_test": {
            "video_dir": "/data0/wzz/data/Charades_STA/videos",
            "ann_file": "/data0/wzz/data/Charades_STA/charades_test.json",
            "feat_file": "/data0/wzz/data/Charades_STA/vgg_rgb_features.hdf5",
        },
        "stvg_train": {
            'feat_roots': ['/path/to/HC-STVG-Files/csn_train_feats', '/path/to/HC-STVG-Files/flow_train_feats'],
            'prepdir': '/path/to/HC-STVG-Files/train_prep',
        },
        "stvg_val":{
            'feat_roots': ['/path/to/HC-STVG-Files/csn_val_feats', '/path/to/HC-STVG-Files/flow_val_feats'],
            'prepdir': '/path/to/HC-STVG-Files/val_prep',
        },
        "stvg_test": {
            'feat_roots': ['/path/to/HC-STVG-Files/csn_test_feats', '/path/to/HC-STVG-Files/flow_test_feats'],
            'prepdir': '/path/to/HC-STVG-Files/test_prep',
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        if "stvg_test" in name:
            args = dict(
                feat_roots=attrs['feat_roots'],
                prepdir=attrs['prepdir'],
            )
        elif "stvg" in name:
            args = dict(
                feat_roots = attrs['feat_roots'],
                prepdir = attrs['prepdir'],
            )
        else:
            args = dict(
                root=os.path.join(data_dir, attrs["video_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                feat_file=os.path.join(data_dir, attrs["feat_file"]),
            )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "charades" in name:
            return dict(
                factory = "CharadesDataset",
                args = args
            )
        elif "stvg_test" in name:
            return dict(
                factory="STVGTestset",
                args=args
            )
        elif "stvg" in name:
            return dict(
                factory = "STVGDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
