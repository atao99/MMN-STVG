from tan.data.datasets.stvg import STVGDataset
from tan.data.datasets.stvgtest import  STVGTestset

# path to video-level annotation file
vanno_file = '/path/to/HC-STVG-Files/HCVG_train.json'
# path to dir of tubes of each video generated in the first stage
tanno_root = '/path/to/HC-STVG-Files/train_tubes_1'
# path to extracted features of each tube in each video
feat_files = ['/path/to/HC-STVG-Files/csn_train_split1.pkl', '/path/to/HC-STVG-Files/flow_train_split1.pkl']
# path to preprocessed features of each tube
feat_roots = ['/path/to/HC-STVG-Files/csn_train_feats', '/path/to/HC-STVG-Files/flow_train_feats']
# path to preprocessed labels of each tube
prepdir = '/path/to/HC-STVG-Files/train_prep'
# fromprep = True : read tube labels and features from prepdir and feat_roots
# fromprep = False : genarate tube labels and features and save them in prepdir and feat_roots
fromprep = False
STVGDataset(vanno_file=vanno_file, tanno_root=tanno_root, feat_files=feat_files,
            feat_roots=feat_roots, prepdir=prepdir, fromprep=fromprep)

# vanno_file='/path/to/HC-STVG-Files/HCVG_query.json'
# tanno_root='/path/to/HC-STVG-Files/test_tubes_2'
# feat_files=['/path/to/HC-STVG-Files/csn_test_split2.pkl', '/path/to/HC-STVG-Files/flow_test_split2.pkl']
# feat_roots=['/path/to/HC-STVG-Files/csn_test_feats', '/path/to/HC-STVG-Files/flow_test_feats']
# prepdir='/path/to/HC-STVG-Files/test_prep'
# fromprep=False
# STVGTestset(vanno_file=vanno_file, tanno_root=tanno_root, feat_files=feat_files,
#             feat_roots=feat_roots, prepdir=prepdir, fromprep=fromprep)
