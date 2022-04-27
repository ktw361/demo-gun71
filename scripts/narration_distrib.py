import os
import os.path as osp
import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from moviepy import editor

from epic_stats import Runner, compute_hand_from_entry


class NarrationRunner(Runner):
    
    def __init__(self, 
                 annot_df='/home/skynet/Zhifan/datasets/epic/EPIC_100_train.pkl',
                 *args, 
                 **kwargs):
        self.annot_df = pd.read_pickle(annot_df)
        super(NarrationRunner, self).__init__(*args, **kwargs)

    def _get_distribution(self, hand_img, side):
        """

        Args:
            hand_img (PIL.Image): 
            side (str): 'left' or 'right'

        Returns:
            _type_: ndarray (33,)
        """
        if hand_img is None:
            return np.zeros(len(self.filt_names))
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
        return probs.squeeze().numpy()
    
    def predict_frame_distribution(self, vid, frame_idx):
        """ predict left and right grasp distribution
        
        Returns:
            (l_dist, r_dist), 
                each is ndarray of shape (33, )
            if no left or right hand, generate dist of zeros
        """
        l_crop, r_crop = self.extract_frame_crop(vid, frame_idx)
        l_dist = self._get_distribution(l_crop, 'left')
        r_dist = self._get_distribution(r_crop, 'right')
        return l_dist, r_dist
            
    def draw_frame_distribution(self, vid, frame_idx, text=None):
        img, l_box, r_box = self.prepare_frame_input(vid, frame_idx)
        l_crop = self._crop_from_box(img, l_box)
        r_crop = self._crop_from_box(img, r_box)
        l_probs = self._get_distribution(l_crop, 'left')
        r_probs = self._get_distribution(r_crop, 'right')

        def _draw_axis(ax, probs):
            ind = probs.argmax()
            grasp = self.filt_names[ind]
            start, step, height = 0.05, 0.02765, 0.02
            y1 = start + step * ind
            y2 = start + step * ind + height
            ax.barh(self.filt_names, probs)
            ax.axvspan(xmin=0, xmax=probs[ind], ymin=y1, ymax=y2, color='red', alpha=1.0)
            ax.legend([grasp])
            return ax
        
        figsize = (10, 10)
        fig = plt.figure(figsize=figsize)
        ax1 = plt.subplot(211); plt.axis('off')
        ax2 = plt.subplot(223)
        ax3 = plt.subplot(224)

        if text is not None:
           ax1.set_title(text)
        ax1 = ax1.imshow(img)
        ax2 = _draw_axis(ax2, l_probs)
        ax3 = _draw_axis(ax3, r_probs)
        plt.tight_layout()
        plt.close()
        return fig

    def generate_dist_gif(self, 
                          nid, 
                          save_dir='./output/narration_grasp', 
                          nframes=20,
                          skip_exist=False):
        """ Generate a gif to disk. 
        """
        # if not osp.exists(save_dir): os.makedirs(save_dir)
        outfile = osp.join(save_dir, f'{nid}.mp4')
        if osp.exists(outfile) and skip_exist:
            return
        vid = '_'.join(nid.split('_')[:2])
        st, ed, narration = self.annot_df[
            self.annot_df.index == nid][['start_frame', 'stop_frame', 'narration']].iloc[0]
        frames = []
        step = max(1, (ed+1-st)//nframes)
        for i in tqdm.trange(st, ed+1, step):
            fig = self.draw_frame_distribution(vid, i, text=narration)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(data)

        clip = editor.ImageSequenceClip(frames, fps=5)
        clip.write_videofile(outfile)
        # clip.write_gif(osp.join(save_dir, f'{nid}.gif'))
        
if __name__ == '__main__':
    runner = NarrationRunner()
    filtered = pd.read_json('/home/skynet/Zhifan/epic_clip_viewer/data/filtered_300.json')
    for i, r in tqdm.tqdm(filtered.iterrows(), total=len(filtered)):
        runner.generate_dist_gif(r.narration_id, nframes=20, skip_exist=True)