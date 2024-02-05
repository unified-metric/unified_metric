from typing import List, Optional, Union, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
import importlib
import argparse
import gc
import math
import os
import random
import time
import json
import numpy as np
import traceback
#from utils import metrics

from tqdm import tqdm
import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, StableDiffusionPipeline, PNDMScheduler, EulerAncestralDiscreteScheduler, SchedulerMixin, LMSDiscreteScheduler, DDIMScheduler, DDIMInverseScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteSchedulerOutput
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import warnings

from scipy.stats import multivariate_normal
import argparse
import shutil
import pandas as pd
import pickle

import torch
from datasets import load_dataset
from utils import *
import requests
from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--total_step', type=int, default=None)
parser.add_argument('--resume', type=int, default=-1)
parser.add_argument('--approx_num', type=int, default=1)
parser.add_argument('--auto', action = 'store_true')
parser.add_argument('--max_iter_num', type=int, default=2)
parser.add_argument('--method', type=str, default='simpsons_3_8')
parser.add_argument('--lambd', type=float, default=1.)
parser.add_argument('--val', action = 'store_true')

args = parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
pipe.enable_xformers_memory_efficient_attention()

text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet
unet.to("cuda", dtype=torch.float32)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.to("cuda")
for param in vae.parameters():
    param.requires_grad = False
vae.requires_grad_(False)
vae.to("cuda", dtype=torch.float32)
unet.eval()
text_encoder.eval()
vae.eval()
image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])
all_images = []
result_dict = {}
all_total = {}
all_jacobian = {}
all_coefficient = {}
all_loglikelihood = {}
rec_num_list = [i for i in range(1,args.max_iter_num+1)]

height = 512
width = 512
callback_steps = 1
guidance_scale = 1
num_inference_steps = args.total_step
num_images_per_prompt = 1
LATENT_SCALE_FACTOR = 8
generator = None
latents = None
negative_prompt = ""

scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, compute_likelihood=False, set_alpha_to_zero=False)

cas = CAS_integrator(args.total_step, scheduler = scheduler)

# 4. Prepare timesteps
device = pipe._execution_device
scheduler.set_timesteps(num_inference_steps, device=device)

timesteps = scheduler.timesteps

if args.val == False:
    #dataset = load_dataset("yuvalkirstain/pickapic_v1_no_images")['test_unique']
    dataset = load_dataset("yuvalkirstain/pickapic_v1", split = 'test_unique',streaming = True)
else:
    dataset = load_dataset("yuvalkirstain/pickapic_v1_no_images")['validation_unique']

def get_coefficient(timestep, scheduler):
    if isinstance(scheduler, DDIMScheduler):
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    else:
        prev_timestep = timestep + scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if (prev_timestep >= 0 and prev_timestep <1000) else scheduler.final_alpha_cumprod

    # beta_prod_t = 1 - alpha_prod_t
    return (((1 - alpha_prod_t) * alpha_prod_t_prev / alpha_prod_t) ** (0.5)) - ((1 - alpha_prod_t_prev) ** (0.5))

coef_list = torch.tensor([get_coefficient(t, scheduler) for t in timesteps]).cuda()
if (coef_list == 1.).any():
    print("fuck")
    exit()


# 5. Prepare latent variables

# scheduler.reset()
num_channels_latents = pipe.unet.in_channels

#metrics = metrics()


#ans_dict = {'hps':[], 'image_reward':[], 'pick_score':[], 'clip_score':[]}
#for i in rec_num_list: ans_dict[f'ddim_{i}'] = []

print(timesteps)

for prompt_idx, data in enumerate(dataset):
    try:
        if args.resume > 0 and prompt_idx < args.resume:
            continue
        prompt = data['caption']
        img_0 = Image.open(BytesIO(data['jpg_0'])).resize((512,512))
        #img_0 = Image.open('/home/jovyan/fileviewer/ChunsanHong/cas/exp/prompt_0/image_0.png')
        img_1 = Image.open(BytesIO(data['jpg_1'])).resize((512,512))
        label = 1 - data['label_0']
        if label == 0.5:
            continue
        #score_dict = metrics.score(prompt, [img_0,img_1])
        score_dict = {}
        for rec in rec_num_list: score_dict[f'ddim_{rec}'] = []
        score_dict['answer'] = label
        score_dict['prompt'] = prompt
        print(prompt)
        result_dict = {f'img_{i}': {rec: {'total':0, 'coef*del':0, 'loglikelihood':0, 'history':{'jacobian':None, 'coef': None}} for rec in rec_num_list} for i in range(2)}
        os.makedirs(f'{args.save_path}/prompt_{prompt_idx}', exist_ok = True)

        pipe.check_inputs(prompt, height, width, callback_steps)
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        with torch.no_grad():
            text_embeddings = pipe._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            text_embeddings_uncond = pipe._encode_prompt(
                "", device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        for image_idx, image in enumerate([img_0,img_1]):
            image.save(f'{args.save_path}/prompt_{prompt_idx}/image_{image_idx}.png')
            image = image_transforms(image)
            image = image.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)

            latents = vae.encode(image).latent_dist.mean#.sample()
            
            latents = latents * 0.18215
            latents = latents.to(device=device, dtype=vae.dtype)
            latents_uncond = latents.clone().detach()

            latents_orig = latents.clone().detach()
            latents_uncond_orig = latents_uncond.clone().detach()

            # with torch.no_grad():
            # timesteps = torch.tensor(np.arange(args.total_step+1) * ((1000 - args.start_t) // args.total_step), dtype=torch.int64) + args.start_t
            #         timesteps[-1] = 999

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
            
            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

            jacobian_trace_list = {key:[torch.tensor(0).cuda()] for key in rec_num_list}
            jacobian_trace_list_uncond = {key:[torch.tensor(0).cuda()] for key in rec_num_list}
            jacobian_trace_list_cond = {key:[torch.tensor(0).cuda()] for key in rec_num_list}
            log_likelihood = {key: torch.zeros(batch_size, device="cuda") for key in rec_num_list}

            inversed_latent = {key: None for key in rec_num_list}
            inversed_latent_uncond = {key: None for key in rec_num_list}

            with torch.no_grad():
                for rec_num in rec_num_list:
                    latents = latents_orig.clone().detach()
                    latents_uncond = latents_uncond_orig.clone().detach()
                    
                    for i, t in enumerate(tqdm(timesteps[1:])):
                        
                        # expand the latents if we are doing classifier free guidance
                        latents = latents.detach()
                        latents_uncond = latents_uncond.detach()
                        
                        latents_default = latents.clone()
                        latents_uncond_default = latents_uncond.clone()

                        log_likelihood[rec_num] = log_likelihood[rec_num].detach()

                        # latents.requires_grad_(True)
                        # latents_uncond.requires_grad_(True)

                        for rec_idx in range(rec_num):
                            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                            latent_model_input_uncond = torch.cat([latents_uncond] * 2) if do_classifier_free_guidance else latents_uncond
                            latent_model_input_uncond = scheduler.scale_model_input(latent_model_input_uncond, t)
                            # predict the noise residual
                            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings.detach()).sample
                            noise_pred_uncond = pipe.unet(latent_model_input_uncond, t, encoder_hidden_states=text_embeddings_uncond.detach()).sample

                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                            
                            if rec_idx == 0:
                                tj_samples = []
                                tj_samples_cond = []
                                tj_samples_uncond = []
                                epsilon = 1e-3
                                for rand_step in range(args.approx_num//4):
                                    rand_eps = torch.randint(low=0, high=2, size=(4, *noise_pred.shape[1:])).float().cuda() * 2 - 1
                                    # rand_eps = torch.randint(low=0, high=2, size=(1, *noise_pred.shape), dtype = torch.float16).cuda() * 2 - 1
                                    # grad_fn_eps_cond =  torch.autograd.grad(noise_pred, latents, grad_outputs=rand_eps, retain_graph=(rand_step<0), is_grads_batched=True)[0]
                                    # grad_fn_eps_uncond =  torch.autograd.grad(noise_pred_uncond, latents_uncond, grad_outputs=rand_eps, retain_graph=(rand_step<0), is_grads_batched=True)[0]
                                    noise_pred_eps = pipe.unet(latent_model_input + epsilon * rand_eps, t, encoder_hidden_states = text_embeddings.expand(4, -1, -1).detach()).sample
                                    noise_pred_uncond_eps = pipe.unet(latent_model_input_uncond + epsilon * rand_eps, t, encoder_hidden_states=text_embeddings_uncond.expand(4, -1, -1).detach()).sample
                                    grad_fn_eps_cond = (noise_pred_eps - noise_pred)/epsilon
                                    grad_fn_eps_uncond = (noise_pred_uncond_eps - noise_pred_uncond)/epsilon
                                    tj_sample = torch.sum((grad_fn_eps_cond - args.lambd * grad_fn_eps_uncond) * rand_eps, dim=tuple(range(1, len(rand_eps.shape)))).detach()
                                    tj_sample_cond = torch.sum((grad_fn_eps_cond) * rand_eps, dim=tuple(range(1, len(rand_eps.shape)))).detach()
                                    tj_sample_uncond = torch.sum((grad_fn_eps_uncond) * rand_eps, dim=tuple(range(1, len(rand_eps.shape)))).detach()
                                    tj_samples.append(tj_sample)
                                    tj_samples_cond.append(tj_sample_cond)
                                    tj_samples_uncond.append(tj_sample_uncond)
                                
                                tj_samples = torch.cat(tj_samples, dim=0)
                                tj_samples_cond = torch.cat(tj_samples_cond, dim=0)
                                tj_samples_uncond = torch.cat(tj_samples_uncond, dim=0)
                                jacobian_trace_list[rec_num].append(tj_samples.mean().detach())
                                jacobian_trace_list_cond[rec_num].append(tj_samples_cond.mean().detach())
                                jacobian_trace_list_uncond[rec_num].append(tj_samples_uncond.mean().detach())
                                del tj_samples
                                latents.requires_grad_(False)
                                latents_uncond.requires_grad_(False)
    
                            latents = scheduler.step(noise_pred, t, latents_default).prev_sample
                            latents_uncond = scheduler.step(noise_pred_uncond, t, latents_uncond_default).prev_sample
                    inversed_latent[rec_num] = latents.detach().clone()
                    inversed_latent_uncond[rec_num] = latents_uncond.detach().clone()
                    jacobian_trace_list[rec_num] = torch.stack(jacobian_trace_list[rec_num])
                    jacobian_trace_list_cond[rec_num] = torch.stack(jacobian_trace_list_cond[rec_num])
                    jacobian_trace_list_uncond[rec_num] = torch.stack(jacobian_trace_list_uncond[rec_num])

            # breakpoint()
            for rec in rec_num_list:
                with torch.no_grad():
                    result_dict[f'img_{image_idx}'][rec]['loglikelihood'] = ((multivariate_gaussian_log_likelihood(inversed_latent[rec]) - args.lambd * multivariate_gaussian_log_likelihood(inversed_latent_uncond[rec]))).cpu().item()
                result_dict[f'img_{image_idx}'][rec]['loglikelihood_cond'] = (multivariate_gaussian_log_likelihood(inversed_latent[rec])).cpu().item()
                result_dict[f'img_{image_idx}'][rec]['loglikelihood_uncond'] = (multivariate_gaussian_log_likelihood(inversed_latent_uncond[rec])).cpu().item()
                result_dict[f'img_{image_idx}'][rec]['coef*del'] = -(jacobian_trace_list[rec][1:-1] * coef_list[1:-1]).sum().item()
                result_dict[f'img_{image_idx}'][rec]['total'] = cas.score(result_dict[f'img_{image_idx}'][rec]['loglikelihood'], jacobian_trace_list[rec].cpu().numpy(), [1, -1], args.method)
                score_dict[f'ddim_{rec}'].append(result_dict[f'img_{image_idx}'][rec]['total'])
                # score_dict[f'img_{image_idx}'][f'ddim_{rec}'] = result_dict[f'img_{image_idx}'][rec]['total']
                result_dict[f'img_{image_idx}'][rec]['history']['jacobian'] = jacobian_trace_list[rec].cpu().tolist()
                result_dict[f'img_{image_idx}'][rec]['history']['jacobian_cond'] = jacobian_trace_list_cond[rec].cpu().tolist()
                result_dict[f'img_{image_idx}'][rec]['history']['jacobian_uncond'] = jacobian_trace_list_uncond[rec].cpu().tolist()
                result_dict[f'img_{image_idx}'][rec]['history']['coef'] = coef_list.cpu().tolist()
        

        with open(f'{args.save_path}/prompt_{prompt_idx}/result_dict.pickle', 'wb') as f:
            pickle.dump(result_dict,f)
        with open(f'{args.save_path}/prompt_{prompt_idx}/score_dict.pickle', 'wb') as f:
            pickle.dump(score_dict,f)
        #for key in ans_dict.keys():
        #    pred = int(score_dict[key][0] < score_dict[key][1])
        #    ans_dict[key].append((pred==score_dict['answer']))
        #    print(f'{key}: {sum(ans_dict[key])/len(ans_dict[key])}')
        breakpoint()
        print(f'total #: {len(ans_dict[key])}')
        # if len(ans_dict[key]) == 100:
        #     break
        # breakpoint()
    except Exception as e:
        print(traceback.format_exc() + str(e))
        pass
