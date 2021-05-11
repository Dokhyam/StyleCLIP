import os
import numpy as np
import torch
from models.stylegan2.model import Generator
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load('mapper/pretrained/stylegan2-ffhq-config-f.pt')
StyleGANGenerator = Generator(1024,512,8).to(device).eval()
StyleGANGenerator.load_state_dict(ckpt['g_ema'], strict=False)

def generate_image_from_latents(latent_code, randomize_noise=True):
	out = StyleGANGenerator([latent_code],input_is_latent=True, randomize_noise=True)
	return out[0].squeeze(0).transpose(0,1).transpose(1,2).detach().cpu().numpy()

def get_clip_text_embeddings(text:list, clip_model):
	tokenized_text = clip.tokenize(text).to(device)
	return clip_model.encode_text(tokenized_text)

def get_clip_image_embeddings(preprocess, image, clip_model):
	preprocessed_image = preprocess(Image.fromarray((255*image).astype(np.uint8))).unsqueeze(0).to(device)
	return clip_model.encode_image(preprocessed_image)

def cosine_similarity(a,b):
	return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

def run():
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	directions_path = '/disk1/dokhyam/StyleCLIP/directions/'
	image_latents = torch.load('/disk1/dokhyam/StyleCLIP/mapper/latent_data/train_faces.pt')
	directions_list = os.listdir(directions_path)
	image_ind = 0
	
	for d_file in directions_list:
		d = torch.load(directions_path + d_file)
		alpha =  np.random.uniform(1.0,4.0)
		input_batch = image_latents[image_ind,:,:]
		input_cuda = input_batch.to(device).float()
		I1 = generate_image_from_latents(input_cuda.unsqueeze(0))
		I2 = generate_image_from_latents(input_cuda.unsqueeze(0) + alpha*d.to(device))
		s_neutral = 'face with hair'
		neutral_embeddings = get_clip_text_embeddings(s_neutral, clip_model)
		s_temp = ['Her hair is in a tight bun', 'She has a brown bob']
		with torch.no_grad():
			image_embeddings_diff = get_clip_image_embeddings(clip_preprocess,I1, clip_model) - get_clip_image_embeddings(clip_preprocess,I2, clip_model)
			text_embeddings_diff = get_clip_text_embeddings(s_temp,clip_model) - neutral_embeddings
			for i in range(len(text_embeddings_diff)):
				cos_sim = cosine_similarity(image_embeddings_diff.cpu().numpy(),text_embeddings_diff[i].cpu().numpy()) 
				print(f'text: {s_temp[i]}. score: {cos_sim}')
			

		image_ind +=1
		
if __name__ == "__main__":
	run()
	
	
	
	

