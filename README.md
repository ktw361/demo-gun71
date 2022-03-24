# Demo for obtaining hand grasp given hand detections

```bash
conda env create --f environment.yml
conda activate humanhands
```

```bash
CUDA_VISIBLE_DEVICES=GPU_ID python demo_handgrasp.py
```

You should get the following output
```
Hand grasp for left hand: Precision Sphere
Hand grasp for right hand: Medium Wrap
```