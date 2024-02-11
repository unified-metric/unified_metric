from utils import *
import torch
import pickle
from PIL import Image
from diffusers import DDIMInverseScheduler
import torch
import argparse
import numpy as np
from datasets import load_dataset
from io import BytesIO
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--total_step', type=int, default=10)
parser.add_argument('--approx_num', type=int, default=20)
parser.add_argument('--rec_num', type=int, default=2)

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, compute_likelihood=False, set_alpha_to_zero=False)
scheduler.set_timesteps(args.total_step)

cas_preprocessor = CAS_preprocessor(model = 'SD1.5', scheduler = scheduler, device = device, num_inference_steps = args.total_step, approx_num = args.approx_num, rec_num = args.rec_num)
cas_integrator = CAS_integrator(args.total_step, scheduler = scheduler)

dataset = load_dataset("yuvalkirstain/pickapic_v1", split = 'test_unique',streaming = True)
total, right = 0, 0

for data in dataset:
    prompt = data['caption']
    label = 1 - data['label_0']
    if label == 0.5: continue
    img_0 = Image.open(BytesIO(data['jpg_0']))
    img_1 = Image.open(BytesIO(data['jpg_1']))
    res_0 = cas_preprocessor.preprocess(img_0, prompt)
    res_1 = cas_preprocessor.preprocess(img_1, prompt)
    cas_0 = cas_integrator.score(res_0['llhood']['total'], res_0['jacobian']['total'])
    cas_1 = cas_integrator.score(res_1['llhood']['total'], res_1['jacobian']['total'])
    pred = int(cas_0 < cas_1)
    total += 1
    if pred == label: right += 1
    print(f'Processed Image #: {total}, Current Accuracy: {right/total:.3f}')
unused_idx = np.array([ 34,  38,  65,  68,  74, 160, 285, 292, 304, 343, 344, 378, 431])