from os.path import join, splitext, exists
from os import listdir, mkdir
import pickle
import json
import logging
import torch
from .utils import moment_to_stiou2d, get_vid_feat, get_tube_feat
from transformers import AutoTokenizer

class STVGTestset(torch.utils.data.Dataset):
    def __init__(self, vanno_file=None,
                 tanno_root=None,
                 feat_files=[],
                 feat_roots=[],
                 num_pre_clips=32,
                 num_clips=16,
                 prepdir='/path/to/prepdir',
                 fromprep=True):
        super(STVGTestset, self).__init__()

        logger = logging.getLogger("tan.tester")
        logger.info("Preparing data, please wait...")

        self.feat_roots = feat_roots
        self.annos = []
        if fromprep:
            logger.info("Preparing data from prepdir:"+prepdir)
            pklfiles = listdir(prepdir)
            for fname in pklfiles:
                fpath = join(prepdir, fname)
                with open(fpath, 'rb') as f:
                    anno = pickle.load(f)
                self.annos.append(anno)
            return

        with open(vanno_file,'r') as f:
            vannos = json.load(f)

        features = []
        for feat_file in feat_files:
            with open(feat_file, 'rb') as f:
                features.append(torch.load(f))

        num_feats = len(feat_files)

        # tokenizer = AutoTokenizer.from_pretrained('albert-xxlarge-v1')
        tokenizer = AutoTokenizer.from_pretrained('./distilbert-base-uncased')

        for feat_root in feat_roots:
            if not exists(feat_root):
                mkdir(feat_root)

        if not exists(prepdir):
            mkdir(prepdir)

        cnt = 0
        ncnt = 0
        for vname, sentence in vannos.items():

            query = tokenizer(sentence, return_tensors="pt")['input_ids']
            word_len = torch.tensor([query.size(-1)])

            vid = splitext(vname)[0]
            tanno_path = join(tanno_root, vid+'_tubes.pkl')
            if not exists(tanno_path):
                ncnt += 1
                continue
            with open(tanno_path,'rb') as f:
                tanno = pickle.load(f)
            # tanno: {0: [tube0_len*6, tube0_score],
            #            [tube1_len*6, tube2_score],
            #            ......}
            # 6 means: frame_num, x1, y1, x2, y2, dector_score
            vfeats = [features[i][vid] for i in range(num_feats)]
            tanno = tanno[0]
            tubes = []
            feats = []
            for i,(tube, _) in enumerate(tanno):
                tube = tube[:,:5]
                tube = torch.Tensor(tube)

                tubes.append(tube)
                tube_feats = []
                for vfeat in vfeats:
                    tube_feat = vfeat[i]
                    tube_feat = torch.stack(tube_feat, dim=0)[:, 1:].float()
                    tube_feat = get_tube_feat(tube_feat, num_pre_clips)
                    tube_feats.append(tube_feat)
                feats.append(tube_feats)

            tubenum = len(tubes)
            if tubenum == 0:
                print(vid)
                continue
            # all_feats = torch.stack(feats)
            all_feats = []
            for n in range(num_feats):
                all_feats.append(torch.stack([feats[i][n] for i in range(tubenum)]))

            anno = {
                'vid': vname,
                'tubes': tubes,  # [(n1, 5), (n2, 5), ...]
                'sentence': sentence,  # raw string
                'query': query,  # (1, word_len) for BERT
                'wordlen': word_len,
                'tubenum': tubenum,
            }
            self.annos.append(anno)

            with open(join(prepdir, vid+'.pkl'), 'wb') as f:
                pickle.dump(anno, f)

            # with open(join(feat_root, vid+'.pkl'), 'wb') as f:
            #     pickle.dump(all_feats, f)

            for i in range(num_feats):
                with open(join(feat_roots[i], vid+'.pkl'), 'wb') as f:
                    pickle.dump(all_feats[i], f)

            cnt += 1
            print(cnt, 'done')

        print('ncnt:', ncnt)
        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="tacos")

    def __getitem__(self, idx):
        # feat = self.feats[self.annos[idx]['vid']]
        # feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="tacos")

        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        num_tubes = self.annos[idx]['tubenum']
        vid = splitext(self.annos[idx]['vid'])[0]
        feats = []
        for feat_root in self.feat_roots:
            with open(join(feat_root, vid + '.pkl'), 'rb') as f:
                feats.append(pickle.load(f))
        feat = torch.cat(feats, dim=2)
        # return feat, query, wordlen, iou2d, moment, idx
        return feat, query, wordlen, num_tubes, idx

    def __len__(self):
        return len(self.annos)

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_vid(self, idx):
        return self.annos[idx]['vid']

    def get_tubes(self, idx):
        return self.annos[idx]['tubes']
