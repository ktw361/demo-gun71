from PIL import Image
from pathlib import Path
import numpy as np


def read_epic_image(video_id, 
                    frame_idx, 
                    root='/home/skynet/Zhifan/data/epic_rgb_frames/',
                    as_pil=False):
    root = Path(root)
    frame = root/video_id[:3]/video_id/f"frame_{frame_idx:010d}.jpg"
    frame = Image.open(frame)
    if as_pil:
        return frame
    else:
        return np.asarray(frame)

    
