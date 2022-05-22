from os.path import join, splitext, exists
from os import listdir, mkdir
import pickle
import json
import logging
import torch
from .utils import moment_to_stiou2d, get_vid_feat, get_tube_feat
from transformers import AutoTokenizer

class STVGDataset(torch.utils.data.Dataset):
    def __init__(self,
                 vanno_file=None,
                 tanno_root=None,
                 feat_files=[],
                 feat_roots=[],
                 num_pre_clips=32,
                 num_clips=16,
                 prepdir='/path/to/prepdir',
                 fromprep=True):
        super(STVGDataset, self).__init__()

        logger = logging.getLogger("tan.trainer")
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

        tokenizer = AutoTokenizer.from_pretrained('./distilbert-base-uncased')
        # tokenizer = AutoTokenizer.from_pretrained('./albert-xxlarge-v1')

        for feat_root in feat_roots:
            if not exists(feat_root):
                mkdir(feat_root)

        if not exists(prepdir):
            mkdir(prepdir)



        cnt = 0
        ncnt = 0
        for vname, vanno in vannos.items():

            vid = splitext(vname)[0]
            tanno_path = join(tanno_root, vid + '_tubes.pkl')
            if not exists(tanno_path):
                ncnt += 1
                continue
            # video-level information
            moment = torch.Tensor([vanno['st_frame'],vanno['ed_frame']])
            bbox_num = torch.arange(moment[0],moment[1]+1).unsqueeze(1)
            bbox = torch.Tensor(vanno['bbox']) # (frames, 4)
            # gttube_full = torch.cat([bbox_num, bbox],dim=1)
            # gttube_full[:, 3] = gttube_full[:, 1] + gttube_full[:, 3]
            # gttube_full[:, 4] = gttube_full[:, 2] + gttube_full[:, 4]
            # start = ((6-(int(vanno['st_frame'])-1)%6))%6
            # gttube = gttube_full[start::6].contiguous()
            gttube = torch.cat([bbox_num, bbox], dim=1)
            gttube[:, 3] = gttube[:, 1] + gttube[:, 3]
            gttube[:, 4] = gttube[:, 2] + gttube[:, 4]
            sentence = vanno['English']

            # tube-level information
            query = tokenizer(sentence, return_tensors="pt")['input_ids']
            word_len = torch.tensor([query.size(-1)])

            with open(tanno_path,'rb') as f:
                tanno = pickle.load(f)
            # tanno: {0: [tube0_len*6, tube0_score],
            #            [tube1_len*6, tube2_score],
            #            ......}
            # 6 means: frame_num, x1, y1, x2, y2, dector_score
            vfeats = [features[i][vid] for i in range(num_feats)]
            tanno = tanno[0]
            tubes = []
            all_iou2d = []
            feats = []
            for i,(tube, _) in enumerate(tanno):
                tube = tube[:,:5]
                tube = torch.Tensor(tube)
                # st_frame = int(tube[0][0].item())
                # ed_frame = int(tube[-1][0].item())
                # tube_full = torch.empty(ed_frame-st_frame+1, 5)
                # tube_full[-1] = tube[-1]
                # for i in range(tube.size(0) - 1):
                #     index = torch.linspace(0, 1, 7).reshape(7, 1)
                #     a = tube[i]
                #     b = tube[i + 1]
                #     c = (b - a).unsqueeze(0)
                #     # subproposal = (torch.matmul(index, c) + a).round().long()
                #     subproposal = torch.matmul(index, c) + a
                #     tube_full[i * 6:(i + 1) * 6 + 1, :] = subproposal
                tubes.append(tube)
                iou2d = moment_to_stiou2d(gttube, tube, num_clips)
                all_iou2d.append(iou2d)
                tube_feats = []
                for vfeat in vfeats:
                    tube_feat = vfeat[i]
                    tube_feat = torch.stack(tube_feat,dim=0)[:,1:].float()
                    tube_feat = get_tube_feat(tube_feat,num_pre_clips)
                    tube_feats.append(tube_feat)
                feats.append(tube_feats)

            tubenum = len(tubes)
            if tubenum == 0:
                print(vid)
                continue
            all_iou2d = torch.stack(all_iou2d)
            all_feats = []
            for n in range(num_feats):
                all_feats.append(torch.stack([feats[i][n] for i in range(tubenum)]))

            anno = {
                'vid': vid,
                'gttube': gttube,  # (num_frames, 5)
                'tubes': tubes,  # [(n1, 5), (n2, 5), ...]
                'iou2d': all_iou2d,  # number_of_tubes*num_clips*num_clips
                'sentence': sentence,  # raw string
                'query': query,  # (1, word_len) for BERT
                'wordlen': word_len,
                'tubenum': tubenum,
            }
            self.annos.append(anno)

            with open(join(prepdir, vid+'.pkl'), 'wb') as f:
                pickle.dump(anno, f)

            for i in range(num_feats):
                with open(join(feat_roots[i], vid+'.pkl'), 'wb') as f:
                    pickle.dump(all_feats[i], f)

            cnt += 1
            print(cnt, 'done')

        print('ncnt:',ncnt)
        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="tacos")

    def __getitem__(self, idx):
        # feat = self.feats[self.annos[idx]['vid']]
        # feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="tacos")

        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        iou2d = self.annos[idx]['iou2d']
        num_tubes = self.annos[idx]['tubenum']
        vid = self.annos[idx]['vid']
        feats = []
        for feat_root in self.feat_roots:
            with open(join(feat_root, vid+'.pkl'), 'rb') as f:
                feats.append(pickle.load(f))
        feat = torch.cat(feats,dim=2)
        # return feat, query, wordlen, iou2d, moment, idx
        return feat, query, wordlen, iou2d, num_tubes, idx

    def __len__(self):
        return len(self.annos)

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_vid(self, idx):
        return self.annos[idx]['vid']

    def get_gttube(self, idx):
        return self.annos[idx]['gttube']

    def get_tubes(self, idx):
        return self.annos[idx]['tubes']



