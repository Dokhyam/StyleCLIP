import os
import datasets
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
import time
from datasets import load_dataset
import transformers
from transformers import DataCollatorForLanguageModeling,LineByLineTextDataset,TextDataset, Trainer, TrainingArguments

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
	return tokenizer(examples["text"])

def group_texts(examples, block_size=128):
	# Concatenate all texts.
	concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
	total_length = len(concatenated_examples[list(examples.keys())[0]])
	# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
		# customize this part to your needs.
	total_length = (total_length // block_size) * block_size
	# Split by chunks of max_len.
	result = {
		k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
		for k, t in concatenated_examples.items()
	}
	result["labels"] = result["input_ids"].copy()
	return result

def train_iteration(
	results_path,
	d_data_path,
	sentences_data_path,
	val_sentences_data_path, 
	saved_models_path,
	with_d=True,
	previous_model_path=None
	):
	if not os.path.exists(saved_models_path):
		os.mkdir(saved_models_path)
	# Training and optimization configs 
	

	if previous_model_path is None:
		gpt2_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id,output_hidden_states=True)
	else:
		gpt2_model = transformers.GPT2LMHeadModel.from_pretrained(previous_model_path, pad_token_id=tokenizer.eos_token_id,output_hidden_states=True)
# 	gpt2_model = gpt2.model.train()
	max_length = 20
	eof = '<|endoftext|>'
	block_size = 128
	# dataloaders
	datasets = load_dataset("text", data_files={"train":sentences_data_path , "validation": val_sentences_data_path})
	tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
	lm_datasets = tokenized_datasets.map(
		group_texts,
		batched=True,
		batch_size=64,
		num_proc=4,
	)

	training_args = TrainingArguments(
	    output_dir=results_path, #The output directory
	    overwrite_output_dir=True, #overwrite the content of the output directory
	    num_train_epochs=30, # number of training epochs
	    per_device_train_batch_size=4, # batch size for training
	    per_device_eval_batch_size=4,  # batch size for evaluation
	    evaluation_strategy = "epoch",
	    save_steps=100, # after # steps model is saved
	    warmup_steps=10,# number of warmup steps for learning rate scheduler
	    prediction_loss_only=True,
	    )

	trainer = Trainer(
	    model=gpt2_model,
	    args=training_args,
	    train_dataset=lm_datasets["train"],
	    eval_dataset=lm_datasets["validation"],
	)
	trainer.train()


if __name__ == "__main__":
	BASE_PATH = '/disk1/dokhyam/Style2Text/'
	d_data_path = BASE_PATH + 'directions/'
	sentences_data_path =  BASE_PATH + 'sentences.txt'
	val_sentences_data_path = BASE_PATH + 'sentences.txt'
	results_path1 = "/home/dokhyam/trainer_out"
	saved_models_path = '/home/dokhyam/Models/'
	train_iteration(
		results_path1,
		d_data_path,
		sentences_data_path,
		val_sentences_data_path,
		saved_models_path
		)
	results_path2 = "/home/dokhyam/trainer_out2"
	train_iteration(
		results_path2,
		d_data_path,
		sentences_data_path,
		val_sentences_data_path,
		saved_models_path,
		previous_model_path=os.path.join(results_path1, 'checkpoint-500')
		)
