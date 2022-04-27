import os
import os.path as osp
from cv2 import merge
import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')

import numpy as np
from pathlib import Path

from epic_stats import Runner

""" Generate grasp table contents

We didn't differentiate left and right hand, 
and use `portable_objects` as criteria to select an IN-CONTACT hands
    
    -------------------------
    cup | take-on   | IMAGE
        |-------------------
        | put-up    | IMAGE
    -------------------------
        
"""

class DetailedRunner(Runner):
    
    def __init__(self, 
                 *args, 
                 **kwargs):
        super(DetailedRunner, self).__init__(*args, **kwargs)

    def collect_all(self,
                    annot_df='/home/skynet/Zhifan/epic_clip_viewer/data/filtered_300.json',
                    num_compute=-1,
                    num_samples_per_clip=3):
        """
        Generate two structures called obj_map and verb_map

        For obj_map:
            - <object>:
                - <verb>: a dict mapping from `grasp` to `num_occurrence`
                    # - 'left': a dict mapping from `grasp` to `num_occurrence`
                    # - 'right': a dict mapping from `grasp` to `num_occurrence`
        
        For verb_map:
            - <verb>:
                - <object>: a dict mapping from `grasp` to `num_occurrence`
                    # - 'left': a dict mapping from `grasp` to `num_occurrence`
                    # - 'right': a dict mapping from `grasp` to `num_occurrence`

        Returns:
            (obj_map, verb_map)
        """
        obj_map = dict()
        verb_map = dict()

        annot_df = pd.read_json(annot_df)

        for i, entry in tqdm.tqdm(annot_df.iterrows(), total=len(annot_df)):
            if num_compute > 0 and i > num_compute:
                break
            vid = entry.video_id
            st, ed = entry.start_frame, entry.stop_frame
            verb = entry.verb
            noun = entry.noun

            vid_df = self._get_vid_df(vid)
            pool = vid_df[(vid_df.frame >= st) & (vid_df.frame <= ed) & (vid_df.hoa_link == 'portable_object')]
            if len(pool) <= 0:
                continue
            for pool_ind in np.random.choice(len(pool), num_samples_per_clip):
                # selected = pool.iloc[np.random.choice(len(pool), 1)]
                selected = pool.iloc[pool_ind]
                frame_idx = int(selected.frame)
                side = str(selected.side)
                l_grasp, r_grasp = self.compute_frame_stat(vid, frame_idx)
                if side == 'left':
                    grasp = l_grasp
                else:
                    grasp = r_grasp
                
                if grasp == 'None':
                    continue 

                # Append into obj_map
                vmap = obj_map.get(noun, dict())
                hist = vmap.get(verb, dict())
                hist[grasp] = hist.get(grasp, 0) + 1
                vmap[verb] = hist
                obj_map[noun] = vmap
                
                omap = verb_map.get(verb, dict())
                hist = omap.get(noun, dict())
                hist[grasp] = hist.get(grasp, 0) + 1
                omap[noun] = hist
                verb_map[verb] = omap
                
        return obj_map, verb_map

    def compute_histogram(self, grasp_map, title=None):
        """

        Args:
            grasp_map (dict): a mapping from grasp to num_occurrence
        
        Return:
            matplotlib figure
        """
        complete = dict()
        for g in self.filt_names:
            complete[g] = grasp_map.get(g, 0)

        fig = plt.figure(figsize=(7, 7))
        plt.barh(list(complete.keys()), list(complete.values()))
        plt.xticks(fontsize=12)
        if title is not None:
            plt.title(title)
        plt.close()
        return fig
    
    def collect_and_save(self,
                         annot_df='/home/skynet/Zhifan/epic_clip_viewer/data/filtered_300.json',
                         save_dir='./output'):
        """
        Save grasp histogram into save_dir as:

        - object_distrib
            - cup 
                - take-on.png
                - put-onto.png
        
        - verb_distrb
            - take-on
                - cup.png
                - mug.png
        """
        obj_map, verb_map = self.collect_all()

        def _save_two_level_hist(mp, root_dir):
            if not osp.exists(root_dir):
                os.makedirs(root_dir)
            for k1, v1 in mp.items():
                tgt1 = Path(root_dir)/k1
                if not osp.exists(tgt1):
                    os.makedirs(tgt1)
                
                merged = dict()
                for k2, v2 in v1.items():
                    tgt2 = tgt1/f'{k2}.png'
                    title = f'{k1} {k2}'
                    fig = self.compute_histogram(v2, title=title)
                    fig.savefig(tgt2, bbox_inches='tight')
                    
                    for g, g_cnt in v2.items():
                        merged[g] = merged.get(g, 0) + g_cnt

                fig = self.compute_histogram(merged, title=f'{k1}')
                fig.savefig(str(tgt1) + '.png', bbox_inches='tight')

        _save_two_level_hist(obj_map, osp.join(save_dir, 'object_distrib'))
        _save_two_level_hist(verb_map, osp.join(save_dir, 'verb_distrib'))

    
if __name__ == '__main__':
    runner = DetailedRunner()
    runner.collect_and_save()