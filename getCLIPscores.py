import os
import torch
from models.stylegan2.model import Generator

directions_path = '/disk1/dokhyam/StyleCLIP/directions/'
image_latents = torch.load('/disk1/dokhyam/StyleCLIP/mapper/latent_data/train_faces.pt')
directions_list = os.listdir(directions_path)
image_ind = 0
StyleGANGenerator = Generator(1024,512,8)
for d in directions_list:
	input_batch = image_latents[i,:]
	input_cuda = input_batch.cuda().float()
	I_1 = StyleGANGenerator(input_cuda)
	
