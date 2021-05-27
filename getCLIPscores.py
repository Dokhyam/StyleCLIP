
import os
import numpy as np
import torch
from models.stylegan2.model import Generator
import clip
from PIL import Image
import imageio as io
import random


device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load('mapper/pretrained/stylegan2-ffhq-config-f.pt')
StyleGANGenerator = Generator(1024,512,8).to(device).eval()
StyleGANGenerator.load_state_dict(ckpt['g_ema'], strict=False)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def generate_image_from_latents(latent_code, randomize_noise=True):
	out = StyleGANGenerator([latent_code],input_is_latent=True, randomize_noise=True)
	return out[0].squeeze(0).transpose(0,1).transpose(1,2).detach().cpu().numpy()

def get_clip_text_embeddings(text:list):
	tokenized_text = clip.tokenize(text).to(device)
	text_features = clip_model.encode_text(tokenized_text)
	text_features = text_features / text_features.norm(dim=-1, keepdim=True)
	return text_features

def get_clip_image_embeddings(preprocess, image):
	preprocessed_image = preprocess(Image.fromarray((255*image).astype(np.uint8))).unsqueeze(0).to(device)
	image_features = clip_model.encode_image(preprocessed_image)
	image_features = image_features / image_features.norm(dim=-1, keepdim=True)
	return image_features

def cosine_similarity(a,b):
	return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

def per_d_function(d,S, image_latents):
	alpha =  np.random.uniform(1.5,5)
	print(alpha)
	cos_sim_all = []
	image_ind = random.randint(0,len(image_latents)-1)
	input_batch = image_latents[image_ind,:,:]
	input_cuda = input_batch.to(device).float()
	I1 = generate_image_from_latents(input_cuda.unsqueeze(0), False)
	I2 = generate_image_from_latents(input_cuda.unsqueeze(0) + alpha*d.to(device),False)
	io.imwrite('/home/dokhyam/I1.jpg',I1)
	io.imwrite('/home/dokhyam/I2.jpg',I2)
	s_neutral = 'A photo of a face with hair'
	neutral_embeddings = get_clip_text_embeddings(s_neutral)
	with torch.no_grad():
		image_embeddings_diff = get_clip_image_embeddings(clip_preprocess,I2) - get_clip_image_embeddings(clip_preprocess,I1)
		image_embeddings_diff = image_embeddings_diff / image_embeddings_diff.norm(dim=-1, keepdim=True)
		text_embeddings_diff = get_clip_text_embeddings(S) - neutral_embeddings
		text_embeddings_diff = text_embeddings_diff / text_embeddings_diff.norm(dim=-1, keepdim=True)
		for i in range(len(text_embeddings_diff)):
			cos_sim = cosine_similarity(image_embeddings_diff.cpu().numpy(),text_embeddings_diff[i].cpu().numpy()) 
			print(f'text: {S[i]}. score: {cos_sim}')
			cos_sim_all.append(cos_sim)
	return cos_sim_all
			

def run():
	
	directions_path = '/disk1/dokhyam/StyleCLIP/mapper/results/mohawk/inference_results/'
	image_latents = torch.load('/disk1/dokhyam/StyleCLIP/mapper/latent_data/train_faces.pt')
	directions_list = os.listdir(directions_path)
	
	for d_file in directions_list:
		if 'latent_delta' in d_file:
			d = torch.load(directions_path + d_file)
			sim = per_d_function(d, ['A photo of a face with short hair', 'A photo of a face with curly hair'],image_latents)
		else:
			continue
		
if __name__ == "__main__":
	run()
	
	
	
	

