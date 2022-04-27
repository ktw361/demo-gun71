import sys
sys.path.append('/home/skynet/Zhifan/homan-full/')
from homan.datasets import epichoa

import pickle
import os
import os.path as osp
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
from model_grasp import ClassificationNet, SimCLRwGUN71

from demo_handgrasp import create_bbox_fromhand, crop_image
from utils import set_torch_seed, load_config
from data_utils import read_epic_image


def compute_hand_from_entry(entry):
    return (entry.left, entry.top, entry.right, entry.bottom)


class Runner:
    
    def __init__(self,
                 epic_root='/home/skynet/Zhifan/data/epic_rgb_frames',
                 hoa_path='/home/skynet/Zhifan/datasets/epic/hoa/',
                 model_path="./data/SimCLRwGUN71_nosup_tsc_seed0_checkpoint_27.pth",
                 grasp_info="./metadata/grasp_info.yaml"
                 ):
        self.box_scale = np.asarray([456, 256, 456, 256]) / [1920, 1080, 1920, 1080]
        self.hand_img_size = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epic_root = epic_root
        self.hoa_path = hoa_path

        hand_classes = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10',
        'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20',
        'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27', 'G28', 'G29', 'G30',
        'G31', 'G32', 'G33', 'G34', 'G35', 'G36', 'G37', 'G38', 'G39', 'G40',
        'G41', 'G42', 'G43', 'G44', 'G45', 'G46', 'G47', 'G48', 'G49', 'G51',
        'G52', 'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G60', 'G62',
        'G63', 'G64', 'G65', 'G66', 'G67', 'G68', 'G69', 'G70', 'G71', 'G72',
        'G73']
        class2id = {j: i for i, j in enumerate(hand_classes)}

        grasp_info = load_config(grasp_info)
        sel_classes = grasp_info['easy_classes']['classes']
        sel_names = grasp_info['easy_classes']['names']

        self.filt_ids = []
        self.filt_names = []
        for c, n in zip(sel_classes, sel_names):
            if c != 'None':
                self.filt_ids.append(class2id[c])
                self.filt_names.append(n)

        best_model = torch.load(model_path)

        if "simclr" in os.path.basename(model_path).lower():
            hand_model = SimCLRwGUN71(nclasses=len(hand_classes), model_config=best_model["config"]['model']).to(self.device)
        else:
            hand_model = ClassificationNet(n_classes=len(hand_classes)).to(self.device)

        mweights = best_model["model_state_dict"]
        hand_model.load_state_dict(mweights)

        # Freeze weights and set the model in eval mode
        hand_model.eval()
        for param in hand_model.parameters():
            param.requires_grad = False
        self.hand_model = hand_model

        self.vid_df = dict()
    
    def _get_vid_df(self, vid):
        if vid in self.vid_df:
            vid_df = self.vid_df[vid]
        else:
            vid_df = epichoa.load_video_hoa(vid, self.hoa_path)
            self.vid_df[vid] = vid_df
        return vid_df
    
    def prepare_frame_input(self, vid, frame_idx):
        """ Retrieve hand bounding boxes of a frame.

        Returns:
            (img, l_box, r_box)
            img: PIL.Image
            l_box, r_box: ndarray of (4,)
        """
        vid_df = self._get_vid_df(vid)
        img = read_epic_image(vid, frame_idx, self.epic_root, as_pil=True)
        entries = vid_df[vid_df.frame == frame_idx]
        if len(entries) == 0:
            return img, None, None
        l_box = None
        r_box = None
        has_left = sum(entries.side == 'left') > 0
        has_right = sum(entries.side == 'right') > 0
        if has_left:
            entry = entries[entries.side == 'left'].iloc[0]
            l_box = compute_hand_from_entry(entry) * self.box_scale
        if has_right:
            entry = entries[entries.side == 'right'].iloc[0]
            r_box = compute_hand_from_entry(entry) * self.box_scale
        return img, l_box, r_box
    
    def _crop_from_box(self, img, box):
        """ 
        Args:
            img: PIL.Image
            box: tuple of (4,)
        
        Returns:
            ndarray of (h, w, 3)
        """
        bbox = create_bbox_fromhand(box, img, inp_fraction=False)
        hand_img = None
        if bbox is not None:
            return crop_image(img, bbox.scale(1.2).expand_to_square())
        else:
            return None
            
    def extract_frame_crop(self, vid, frame_idx):
        """ Given a frame, extract its hand crops

        Args:
            vid (_type_): _description_
            frame_idx (_type_): _description_
        
        Return:
            (l_crop, r_crop): PIL.Image
        """
        img, l_box, r_box = self.prepare_frame_input(vid, frame_idx)
        l_crop = self._crop_from_box(img, l_box)
        r_crop = self._crop_from_box(img, r_box)
        return l_crop, r_crop

    def get_frame_grasp(self, img, l_hand, r_hand):
        """ Given 

        Args:
            img (PIL.Image): 
            l_hand (tuple): (left/x0, top/y0, right/x1, bottom/y1)
            r_hand (tuple): same as l_hand
        
        Returns:
            l_grasp, r_grasp: str
        """
        l_hand_img = self._crop_from_box(img, l_hand)
        r_hand_img = self._crop_from_box(img, r_hand)

        l_grasp, l_score = 'None', 0
        if l_hand_img is not None:
            l_grasp, l_score = self.process_hand_img(l_hand_img, 'left')

        r_grasp, r_score = 'None', 0
        if r_hand_img is not None:
            r_grasp, r_score = self.process_hand_img(r_hand_img, 'right')

        return l_grasp, r_grasp, l_score, r_score

    def compute_frame_stat(self, vid: str, frame_idx: int):
        """_summary_

        Returns:
            l_grasp, r_grasp: str
        """
        if vid in self.vid_df:
            vid_df = self.vid_df[vid]
        else:
            vid_df = epichoa.load_video_hoa(vid, self.hoa_path)
            self.vid_df[vid] = vid_df
        vid_df = self.vid_df[vid]
        img = read_epic_image(vid, frame_idx, self.epic_root, as_pil=True)
        entries = vid_df[vid_df.frame == frame_idx]
        if len(entries) == 0:
            return 'None', 'None'
        l_hand = None
        r_hand = None
        has_left = sum(entries.side == 'left') > 0
        has_right = sum(entries.side == 'right') > 0
        if has_left:
            entry = entries[entries.side == 'left'].iloc[0]
            l_hand = compute_hand_from_entry(entry) * self.box_scale
        if has_right:
            entry = entries[entries.side == 'right'].iloc[0]
            r_hand = compute_hand_from_entry(entry) * self.box_scale
        l_grasp, r_grasp, l_sc, r_sc = self.get_frame_grasp(img, l_hand, r_hand)
        if l_grasp is None: l_grasp = 'None'
        if r_grasp is None: r_grasp = 'None'
        return l_grasp, r_grasp
    
    def compute_vid_stats(self, vid, samples=None, num_samples=0, use_tqdm=False):
        if vid in self.vid_df:
            vid_df = self.vid_df[vid]
        else:
            vid_df = epichoa.load_video_hoa(vid, self.hoa_path)
            self.vid_df[vid] = vid_df
            
        vid_df = self.vid_df[vid]
        st = min(vid_df.frame)
        ed = max(vid_df.frame)
        result = dict()
        if samples is not None:
            loop = samples
        else:
            loop = np.arange(st, ed+1)
        if num_samples > 0:
            loop = np.random.choice(np.arange(st, ed+1), num_samples, replace=False)
        if use_tqdm:
            loop = tqdm.tqdm(use_tqdm)
        for frame_idx in loop:
            entries = vid_df[vid_df.frame == frame_idx]
            if len(entries) == 0:
                continue
            img = read_epic_image(vid, frame_idx, self.epic_root, as_pil=True)
            l_hand = None
            r_hand = None
            has_left = sum(entries.side == 'left') > 0
            has_right = sum(entries.side == 'right') > 0
            if has_left:
                entry = entries[entries.side == 'left'].iloc[0]
                l_hand = compute_hand_from_entry(entry) * self.box_scale
            if has_right:
                entry = entries[entries.side == 'right'].iloc[0]
                r_hand = compute_hand_from_entry(entry) * self.box_scale
            l_grasp, r_grasp, l_sc, r_sc = self.get_frame_grasp(img, l_hand, r_hand)
            if l_grasp is None: l_grasp = 'None'
            if r_grasp is None: r_grasp = 'None'
            key = f"{vid}_{frame_idx}"
            result[key] = (l_grasp, r_grasp, l_sc, r_sc)
        
        return result

    def process_hand_img(self, hand_img, side):
        hand_transform = TF.Compose([
                TF.Resize((self.hand_img_size, self.hand_img_size)),
                TF.ToTensor(),
            ])

        normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        if side == 'left':
            hand_img = TFF.hflip(hand_img)
        hnd = hand_transform(hand_img)
        tensor = normalize(hnd).unsqueeze(0).to(self.device)
        uprobs = self.hand_model.forward_classifier(tensor)[:, self.filt_ids]
        probs = F.softmax(uprobs, dim=1).cpu()
        grasp_ind = probs.argmax(dim=1)[0]
        grasp = self.filt_names[grasp_ind]
        return grasp, probs.max().item()



def run_all_videos():
    rgb_root = '/home/skynet/Zhifan/data/epic_rgb_frames'
    runner = Runner()
    pids = os.listdir(rgb_root)
    final = dict()
    # for pid in sorted(pids):
    NUM_P = 20
    NUM_VID = 20
    pbar = tqdm.tqdm(total=NUM_P * NUM_VID)
    for pid in sorted(np.random.choice(pids, NUM_P)):
        if pid[0] != 'P':
            continue
        ppath = osp.join(rgb_root, pid)
        for vid in sorted(np.random.choice(os.listdir(ppath), NUM_VID)):
            if vid == 'P01_109':
                continue
            pbar.update(1)
            # print(vid)
            try:
                result = runner.compute_vid_stats(vid, num_samples=100)
                final.update(result)
            except OSError:
                print(vid)
            # final[vid] = result
    pbar.close()
    with open('sampled.pkl', 'wb') as fp:
        pickle.dump(final, fp)
    

def get_frame_and_grasp(df, side, topn=1):
    res = []
    if side == 'left':
        argcol = 2
    else:
        argcol = 3
    rdf = df.sort_values(by=argcol, ascending=False).head(topn)
    for frame, ent in rdf.iterrows():
        print(ent)
        lg, rg, ls, rs = ent
        vid = '_'.join(frame.split('_')[:2])
        frame = int(frame.split('_')[-1])
        frame = read_epic_image(vid, frame)
        if side == 'left':
            res.append((frame, lg, ls))
        else:
            res.append((frame, rg, rs))
    return res
            
if __name__ == '__main__':
    runner = Runner()
    # out = runner.compute_vid_stats('P05_08')
    run_all_videos()
    print("Done")