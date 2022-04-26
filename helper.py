import os
import argparse
import git
import yaml, glob
import os.path as osp
from PIL import Image
import numpy as np

# Get path to git repo
def get_git_root(path):
        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root

# Limit CPU Usage
def set_numpythreads():
    os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
    # export VECLIB_MAXIMUM_THREADS=4
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

def str2bool(v):
	if isinstance(v, bool):
			return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
	else:
			raise argparse.ArgumentTypeError('Boolean value expected.')


class GUNReader:
    def __init__(self):
        path = '/home/skynet/Zhifan/demo-gun71/metadata/grasp_info.yaml'
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.graspinfo = dict(zip(config['easy_classes']['names'], config['easy_classes']['classes']))

    def read_grasp(self, s):
        subj = np.random.choice(glob.glob('/home/skynet/Zhifan/data/GRASP/*'), 1)[0]
        img = np.random.choice(
            glob.glob(osp.join(subj, self.graspinfo[s], '*.jpg')), 1)[0]
        return img, Image.open(img)
