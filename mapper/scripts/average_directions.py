import os
import torch
fol_n = '/disk1/dokhyam/StyleCLIP/mapper/results/inference_results/'
dir_f_names = os.listdir(fol_n)
save_dir = '/disk1/dokhyam/StyleCLIP/directions/'
if not os.path.exists(save_dir):
  os.mkdir(save_dir)
counter = 0
delta_avg = 0
direction_count +=1
for f_n in dir_f_names:
  if not 'latents_delta' in f_n:
    continue
  delta_avg += torch.load(os.path.join(fol_n,f_n))
  counter+=1
  if counter ==9:
    torch.save(os.path.join(save_dir,str(direction_count)+'.pt'))
    delta_avg = 0
    counter = 0
    direction_count += 1
