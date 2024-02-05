import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from scipy.integrate import simps
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline, PNDMScheduler
from scipy.signal import savgol_filter
import numpy as np
from typing import *
from torchvision import transforms
from copy import deepcopy


def simpsons_1_3_from_list(mylist):
    return simps(mylist)

def simpsons_3_8_from_list(mylist):
    n = len(mylist) - 1
    if n < 3 or n % 3 != 0:
        raise ValueError("For Simpson's 3/8 rule, length of mylist should be 1 modulo 3 (like 4, 7, 10, ...)")
    integral = mylist[0] + mylist[n]
    for i in range(1, n, 3):
        integral += 3 * (mylist[i] + mylist[i+1])
    for i in range(3, n-2, 3):
        integral += 2 * mylist[i]
    integral *= 3/8
    return integral


def multivariate_gaussian_log_likelihood(x):

    # Calculate the log likelihood for each observation in the batch
    num_samples = x.shape[0]
    log_likelihoods = -0.5 * x.view(num_samples, -1).pow(2).sum(1)

    return log_likelihoods


class CAS_preprocessor(object):
    def __init__(self, model = 'SD1.5', dtype = torch.float32, device = 'cuda', scheduler = None, approx_num = 1, num_inference_steps = 50, epsilon = 1e-3, rec_num = 1):
        self.pipe, self.vae, self.unet, self.text_encoder = {}, {}, {}, {}
        self.dtype = dtype
        self.device = device
        self.scheduler = scheduler
        if model == 'SD1.5': self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = dtype, safety_checker = None).to(device)
        self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.vae = self.pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.unet = self.pipe.unet
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.unet.to(memory_format=torch.channels_last)
        self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])
        self.approx_num = approx_num
        self.num_inference_steps = num_inference_steps
        self.epsilon = epsilon
        self.rec_num = rec_num

    def preprocess(self, image, prompt):
        height = self.unet.config.sample_size * self.pipe.vae_scale_factor
        width = self.unet.config.sample_size * self.pipe.vae_scale_factor
        timesteps = self.scheduler.timesteps[1:]
        with torch.no_grad():
            text_embeddings = {}
            text_embeddings['cond'] = self.pipe._encode_prompt(prompt, self.device, 1, False, "")
            text_embeddings['uncond'] = self.pipe._encode_prompt("", self.device, 1, False, "")
            image = self.image_transforms(image.resize((height,width)))
            image = image.unsqueeze(0).to(device = self.vae.device, dtype = self.vae.dtype)

            init_latents = self.vae.encode(image).latent_dist.mean
            latents = init_latents * 0.18215
            latents = {'cond': latents, 'uncond': latents.clone().detach()}
            jacobian_trace_list = {key: [torch.zeros(self.approx_num).cuda()] for key in ['total', 'cond', 'uncond']}
            
            for t in tqdm(timesteps):
                rand_eps = torch.randint(low=0, high=2, size=(self.approx_num, *init_latents.shape[1:]), dtype = self.dtype).cuda() * 2 - 1
                grad_fn_eps = {}
                latents_default = deepcopy(latents)
                for rec_iter in range(self.rec_num):
                    for preprocess_type in latents.keys():
                        latent_model_input = self.scheduler.scale_model_input(latents[preprocess_type], t)
                        noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[preprocess_type].detach()).sample
    
                        if rec_iter == 0:
                            latents_eps = latents[preprocess_type].clone() + self.epsilon * rand_eps
                            latent_eps_model_input = self.scheduler.scale_model_input(latents_eps, t)
                            noise_pred_eps = self.pipe.unet(latent_eps_model_input, t, encoder_hidden_states = text_embeddings[preprocess_type].expand(self.approx_num, -1, -1)).sample
                        
                            grad_fn_eps[preprocess_type] = (noise_pred_eps - noise_pred) / self.epsilon
                        latents[preprocess_type] = self.scheduler.step(noise_pred, t, latents_default[preprocess_type]).prev_sample

                for preprocess_type in ['total', 'cond', 'uncond']:
                    if preprocess_type == 'total':
                        tj_sample = torch.sum((grad_fn_eps['cond'] - grad_fn_eps['uncond']) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                    else:
                        tj_sample = torch.sum((grad_fn_eps[preprocess_type]) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                    jacobian_trace_list[preprocess_type].append(tj_sample)

            for preprocess_type in ['total', 'cond', 'uncond']:
                jacobian_trace_list[preprocess_type] = torch.cat(jacobian_trace_list[preprocess_type]).reshape(-1,self.approx_num)

        res_dict = {'jacobian': {}, 'llhood': {}}
        
        for preprocess_type in ['cond', 'uncond']:
            res_dict['jacobian'][preprocess_type] = jacobian_trace_list[preprocess_type].cpu().tolist()
            res_dict['llhood'][preprocess_type] = multivariate_gaussian_log_likelihood(latents[preprocess_type].cpu()).item()
        
        res_dict['jacobian']['total'] = jacobian_trace_list['total'].cpu().tolist()
        res_dict['llhood']['total'] = res_dict['llhood']['cond'] - res_dict['llhood']['uncond']
        return res_dict

class CAS_integrator(object):
    def __init__(self, num_timesteps, scheduler = None):
        self.set_config(num_timesteps, scheduler = scheduler)
    def get_coef_list(self):
        alphas = np.array([self.scheduler.alphas_cumprod[(1000//self.num_timesteps)*i].item() for i in range(self.num_timesteps)])
        coef_list = (1/2/alphas/(1-alphas)**(0.5))[:-1]*(alphas[1:]-alphas[:-1])
        return coef_list
    def set_config(self, num_timesteps, scheduler = None):
        self.num_timesteps = num_timesteps
        if scheduler != None:
            self.scheduler = scheduler
        else:
            scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, compute_likelihood=False, set_alpha_to_zero=False)
            scheduler.set_timesteps(num_timesteps)
            self.scheduler = scheduler
        self.timesteps = scheduler.timesteps
        self.coef_list = self.get_coef_list()
    def score(self, log_likelihood, jacobian_list, jacobian_range = [1,-1], method = 'simpsons_3_8'):
        if isinstance(jacobian_list, list):
            jacobian_list = np.array(jacobian_list)
        if len(jacobian_list.shape) == 2:
            jacobian_list = jacobian_list.mean(axis = 1)
        vals = -jacobian_list[:-1]*self.coef_list
        vals = vals[jacobian_range[0]: jacobian_range[1]]
        if method == 'simpsons_1_3':
            return log_likelihood + simps(vals)
        elif method == 'simpsons_3_8':
            return log_likelihood + simpsons_3_8_from_list(vals)