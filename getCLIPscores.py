import os
import torch
from models.stylegan2.model import Generator

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

ckpt = torch.load('mapper/pretrained/stylegan2-ffhq-config-f.pt')
directions_path = '/disk1/dokhyam/StyleCLIP/directions/'
image_latents = torch.load('/disk1/dokhyam/StyleCLIP/mapper/latent_data/train_faces.pt')
directions_list = os.listdir(directions_path)
image_ind = 0
StyleGANGenerator = Generator(64,512,8)
StyleGANGenerator.cuda()
StyleGANGenerator.eval()
StyleGANGenerator.load_state_dict(ckpt['g_ema'], strict=False)
for d in directions_list:
	input_batch = image_latents[image_ind,:,:]
	input_cuda = input_batch.cuda().float()
	out1= StyleGANGenerator([input_cuda.unsqueeze(0)],input_is_latent=True)
	I1 = out1[0].squeeze(0).transpose(0,1).transpose(1,2).detach().cpu().numpy()

