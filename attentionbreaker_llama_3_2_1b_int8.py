import warnings, torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList,  AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

import os
import torch
from datasets import load_dataset
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    QuantoConfig, 
    AwqConfig,
    TrainingArguments,
    pipeline,
)
from bitsandbytes.optim import Adam8bit
import auto_gptq
import torch.nn as nn
import warnings

import numpy as np
import time, os, tqdm, json
import pandas as pd
import copy
import gc
from huggingface_hub import notebook_login

import matplotlib.pyplot as plt
import itertools

import math

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

TASKS = ['abstract_algebra', 
         'anatomy', 
         'astronomy', 
         'business_ethics', 
         'clinical_knowledge', 
         'college_biology', 
         'college_chemistry', 
         'college_computer_science', 
         'college_mathematics', 
         'college_medicine', 
         'college_physics', 
         'computer_security', 
         'conceptual_physics', 
         'econometrics', 
         'electrical_engineering', 
         'elementary_mathematics', 
         'formal_logic', 
         'global_facts', 
         'high_school_biology', 
         'high_school_chemistry', 
         'high_school_computer_science', 
         'high_school_european_history', 
         'high_school_geography', 
         'high_school_government_and_politics', 
         'high_school_macroeconomics', 
         'high_school_mathematics', 
         'high_school_microeconomics', 
         'high_school_physics', 
         'high_school_psychology', 
         'high_school_statistics', 
         'high_school_us_history', 
         'high_school_world_history', 
         'human_aging', 
         'human_sexuality', 
         'international_law', 
         'jurisprudence', 
         'logical_fallacies', 
         'machine_learning', 
         'management', 
         'marketing', 
         'medical_genetics', 
         'miscellaneous', 
         'moral_disputes', 
         'moral_scenarios', 
         'nutrition', 
         'philosophy', 
         'prehistory', 
         'professional_accounting', 
         'professional_law', 
         'professional_medicine', 
         'professional_psychology', 
         'public_relations', 
         'security_studies', 
         'sociology', 
         'us_foreign_policy', 
         'virology', 
         'world_religions'
         ]

# Choices in MMLU
choices = ["A", "B", "C", "D"]

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_name, mode='int8'):
    # Configure BitsAndBytes for int8 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Set for 8-bit loading
        # bnb_8bit_quant_type=mode,  # Keep 'int8' mode (change to your preferred quantization method if any)
        # bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation, can change if needed
        # bnb_8bit_use_double_quant=True,  # Double quantization may improve performance
    )
    
    model = AutoModelForCausalLM.from_pretrained( 
        model_name,
        device_map=device,  # Automatically map model layers to devices
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  # float16 is standard for compute with quantization
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def calculate_perplexity(model, tokenizer, dataset, size):
    total_loss = 0
    total_length = 0
    i = 0#random.randrange(0, len(dataset['text'])-size)
    for example in dataset['text'][i:size+i]:
        if example != '':
            input_text = example
            inputs = tokenizer(input_text, return_tensors="pt")
            print(inputs)
            # print(input_text.dtype, '/',inputs)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss = torch.nan_to_num(loss)
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_length += inputs["input_ids"].size(1)
            # print(example,'/', loss.item(), total_loss, total_length)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_length))
    print(f"WikiText Perplexity: {perplexity:.4f}")
    return perplexity.item()

# Accuracy calculation
def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            # print("pred: %s, gold: %s" % (pred, gold))
            if gold == pred.replace(' ', ''): acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))
    return total_acc/total_num
 

# Format prompt subject 
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

# Format prompts
def format_example(df, idx, subject, include_answer=True):
    prompt = "The following are multiple choice questions (with answers) about {}. ANSWER SHOULD BE IN ANY ONE OF A, B, C OR D. DO NOT ANSWER ANYTHING ELSE. THE ANSWER SHOULD ONLY BE A LETTER AND NOT A NUMBER\n".format(format_subject(subject))
    prompt += df.iloc[idx, 0]
    k = len(df['choices'].iloc[idx])
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df['choices'].iloc[idx][j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df['answer'].iloc[idx]])
    return prompt
 

# Generate prompts from dataset
def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}. ANSWER SHOULD BE IN ANY ONE OF A, B, C OR D. DO NOT ANSWER ANYTHING ELSE. THE ANSWER SHOULD ONLY BE A LETTER AND NOT A NUMBER\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, subject)
    return prompt
 
 
# Tokenize 
def prepare_input(tokenizer, prompts):
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
 
    return input_tokens
 
# Split to different batches 
def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

# Inference of split batches 
def batch_infer(model, tokenizer, prompts):
    batch_size = 1
    answers = []
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        # print(batch_input)
        encode_inputs = prepare_input(tokenizer, batch_input)
        # print(f'{encode_inputs=}')
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=128001, labels = encode_inputs['input_ids'])
        # print(f'{outputs=}')
        answers.append(tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[-1])
        # print('answers: ', answers)
        # break
    # answers = [answer[-1] for answer in answers]
    return answers

# Inference for gradient calculation
def batch_infer_bp_loss(model, tokenizer, prompts, optimizer):
    batch_size = 4
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        encode_inputs = prepare_input(tokenizer, batch_input)
        # print(encode_inputs)
        # st = time.time()
        outputs = model(**encode_inputs, labels=encode_inputs['input_ids'])
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        break

# Zeroth order loss calculation and logits
def batch_infer_zo_loss(model, tokenizer, prompts, optimizer):
    batch_size = 4
    accum_loss = 0
    i=0
    batch_prompts = batch_split(prompts, batch_size)
    for batch_input in batch_prompts:
        encode_inputs = prepare_input(tokenizer, batch_input)
        # print(encode_inputs)
        with torch.no_grad(): 
            outputs = model(**encode_inputs, labels=encode_inputs['input_ids'])
            # outputs = model(**encode_inputs)
        loss = outputs.loss
        i=i+1
        accum_loss = accum_loss+loss.item()
        break
    return loss.item(), outputs.logits

# Calculate MMLU Accuracy
def mmlu_test(model, tokenizer, file_name: str, TASKS):
   
    run_results = {}
    output_filename = 'run_results_%s.json' % (file_name)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        # dev_df = pd.read_csv(os.path.join("mmlu_data/", "dev", task + "_dev.csv"), header=None)[:5]    # Path to MMLU dataset as CSV
        # test_df = pd.read_csv(os.path.join("mmlu_data/", "test", task + "_test.csv"), header=None)     # Path to MMLU dataset as CSV
        
        splits = {'test': task+'/test-00000-of-00001.parquet', 'validation': task+'/validation-00000-of-00001.parquet', 'dev': task+'/dev-00000-of-00001.parquet'}
        dev_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["dev"])[:5]
        test_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
        # print(test_df)
        # break
        for i in range(test_df.shape[0]): # Change the number of iterations to limit the number of prompts for a particular task
            # get prompt and make sure it fits
            k = 5
            prompt_end = format_example(test_df, i, task, include_answer=False)
            # print(prompt_end)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            print(prompt)     # Uncomment to print the prompts provided
            label = choices[test_df['answer'].iloc[i]]
            records.append({'prompt':prompt, 'answer':label})
 
        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
        
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
        print(run_results)    # Uncomment to print run results
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
   
    accuracy = compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    return accuracy

# Get MMLU loss
def mmlu_loss(model, tokenizer, optimizer, file_name: str, TASKS, mode = 'zo'):
   
    run_results = {}
    output_filename = 'run_results_%s.json' % (file_name)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        # dev_df = pd.read_csv(os.path.join("mmlu_data/", "dev", task + "_dev.csv"), header=None)[:5]     # Path to MMLU dataset as CSV
        # test_df = pd.read_csv(os.path.join("mmlu_data/", "test", task + "_test.csv"), header=None)      # Path to MMLU dataset as CSV
        splits = {'test': task+'/test-00000-of-00001.parquet', 'validation': task+'/validation-00000-of-00001.parquet', 'dev': task+'/dev-00000-of-00001.parquet'}
        dev_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["dev"])[:5]
        test_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
        # print(test_df)
        # break
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = 5
            prompt_end = format_example(test_df, i, task, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = choices[test_df['answer'].iloc[i]]
            records.append({'prompt':prompt, 'answer':label})
        
        if mode=='zo':
            loss, logits = batch_infer_zo_loss(model, tokenizer, [record['prompt'] for record in records], optimizer)
            return loss, logits
        elif mode =='bp':
            loss = batch_infer_bp_loss(model, tokenizer, [record['prompt'] for record in records], optimizer)
        
        return loss
    
#Function to flip bits
# BFLIP
def flip_bits_in_tensor(tensor, bit_position):
    bit_mask = 1 << bit_position
    flipped_tensor = tensor ^ bit_mask
    return flipped_tensor.to(torch.int8)

# Functions to load state dictionary
def custom_load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            # print(name)
            if 'weight_format' not in name:
                model_state_dict[name].copy_(param)
 
def custom_load_state_dict_single_layer(model, state_dict, layer_name):
    model_state_dict = model.state_dict()
    model_state_dict[layer_name].copy_(state_dict[layer_name].to(torch.int8))

# Function to calculate importance score using min-max normalization

#SSCORE
def importance_score(w, g, alpha=0.5):
    w_abs = w.detach().abs()
    w_min, w_max = w_abs.min(), w_abs.max()
    w_norm = (w_abs - w_min) / (w_max - w_min + 1e-8)  # Avoid division by zero

    # Normalize g (min-max normalization) in-place
    g_abs = g.detach().abs()
    g_min, g_max = g_abs.min(), g_abs.max()
    g_norm = (g_abs - g_min) / (g_max - g_min + 1e-8)  # Avoid division by zero

    # Compute score in a memory-efficient way using in-place operations
    score = (alpha * g_norm) + ((1 - alpha) * w_norm)

    return score

# Function to calculate importance score using gradient weighted normalization
def grad_weighted_mag_imp_score(w, g):
    # Normalize w (min-max normalization) in-place
    w_abs = w.detach().abs()
    w_min, w_max = w_abs.min(), w_abs.max()
    w_norm = (w_abs - w_min) / (w_max - w_min + 1e-8)  # Avoid division by zero

    # Normalize g (min-max normalization) in-place
    g_abs = g.detach().abs()
    g_min, g_max = g_abs.min(), g_abs.max()
    g_norm = (g_abs - g_min) / (g_max - g_min + 1e-8)  # Avoid division by zero

    # Compute score in a memory-efficient way using in-place operations
    score = g_norm * w_norm
    return score

# Function to calculate importance score using sum-normalization
def sum_norm_importance_score(w, g, alpha=0.5):
    w_abs = w.detach().abs()
    w_norm = w_abs/np.sum(w_abs.cpu().numpy())
    g_abs = g.detach().abs()
    g_norm = g_abs/np.sum(g_abs.cpu().numpy())  
    score = (alpha * g_norm) + ((1 - alpha) * w_norm)
 
    return score

def sensitivity_study(model, tokenizer, optimizer, saved_state_dict, gradients):
    clear_memory()

    # Sensitivity analysis on a larger set of weights

    percent_of_weights = 5
    zo_eps = 1e-3
    custom_load_state_dict(model, saved_state_dict)
    layer_sensitivity = []

    attack_args = {'idx' : [0], 'attack_bit' : 7}

    for name, param in model.named_parameters():
        if 'weight' in name:
            if gradients[name] is not None and param.dtype==torch.int8:
                clear_memory()
                sensitivity = [name]
                orig_data = copy.deepcopy(model.state_dict()[name].data.detach())
                wf1 = torch.flatten(orig_data)
                orig_dtype = wf1.dtype
                k_top =  int((percent_of_weights/100)*gradients[name].detach().view(-1).size()[0])
                w = model.state_dict()[name].data.detach().reshape(-1)
                g = gradients[name].float().detach().reshape(-1)
                imp_score = importance_score(w, g, alpha=0.5)
            
                wval, w_idx = w.detach().abs().topk(k_top)  # topk weights by magnitude
                gval, g_idx = gradients[name].detach().abs().reshape(-1).topk(k_top)    # topk weights by gradients
                ival,i_idx  =  imp_score.topk(k_top)    # topk weights by importance score

                clear_memory()
                state_dict_copy = copy.deepcopy(saved_state_dict)

                # Flip by weight magnitudes
                w[w_idx] = flip_bits_in_tensor(w[w_idx], 7)
                if param.dtype ==torch.int8:
                    state_dict_copy[name].data[:] = w.clone().detach().reshape(param.data.shape)[:] 
                else:
                    state_dict_copy[name].data[:] = w.reshape(param.data.shape)[:]
                custom_load_state_dict_single_layer(model, state_dict_copy, name)

                l,p= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'], mode='zo'), calculate_perplexity(model,tokenizer,wiki_data,size=2)
                sensitivity.append(l[0])
                sensitivity.append(p)

                print(name, "Magnitude based:",l[0],p)

                # Restore to original
                custom_load_state_dict_single_layer(model, saved_state_dict, name)

                clear_memory()

                # Flip by gradients
                if param.dtype ==torch.int8:
                    w = param.data.detach().view(-1).to(torch.int8) 
                else:
                    w = param.data.half().detach().view(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                if param.dtype ==torch.int8:
                    state_dict_copy[name].data[:] = w.clone().detach().reshape(param.data.shape)[:] 
                else:
                    state_dict_copy[name].data[:] = w.reshape(param.data.shape)[:]
                custom_load_state_dict_single_layer(model, state_dict_copy, name)

                l,p= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'], mode='zo'), calculate_perplexity(model,tokenizer,wiki_data,size=2)
                sensitivity.append(l[0])
                sensitivity.append(p)

                print(name, "Gradient based:",l[0],p)

                # Restore to original state
                custom_load_state_dict_single_layer(model, saved_state_dict, name)
            
                clear_memory()

                # Flip bits by importance score
                if param.dtype ==torch.int8:
                    w = param.data.detach().view(-1).to(torch.int8) 
                else:
                    w = param.data.half().detach().view(-1)
                state_dict_copy = copy.deepcopy(saved_state_dict)
                w[i_idx] = flip_bits_in_tensor(w[i_idx], 7)
                if param.dtype ==torch.int8:
                    state_dict_copy[name].data[:] = w.clone().detach().reshape(param.data.shape)[:] 
                else:
                    state_dict_copy[name].data[:] = w.reshape(param.data.shape)[:]
                custom_load_state_dict_single_layer(model, state_dict_copy, name)

                l,p= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'], mode='zo'), calculate_perplexity(model,tokenizer,wiki_data,size=2)
                sensitivity.append(l[0])
                sensitivity.append(p)
                print(name, "Gradient+magnitude based:",l[0],p)
            
                # Restore to original state
                custom_load_state_dict_single_layer(model, saved_state_dict, name) #revert changes
                clear_memory()
                layer_sensitivity.append(sensitivity)

    return layer_sensitivity

def create_population_space(model, gradients, top_5_layers):
    solution_space = []
    for name, param in model.named_parameters():
        if param.dtype==torch.int8 and gradients[name] is not None and name in top_5_layers['Layer'].unique():
            temp_df = top_5_layers[top_5_layers['Layer'] == name]
            k_top =  temp_df['number_of_weights_flipped'].item()
            print(name, k_top)
            if param.dtype==torch.int8:
                w = param.data.detach().clone().view(-1)
            else:
                w = param.data.float().detach().view(-1)
            g = gradients[name].float().detach().view(-1)
 
            imp_score = importance_score(w, g, alpha=0.5)
 
            if temp_df['Attack_type'].item() == 'Magnitude':
                _, idx = w.detach().abs().topk(k_top)
            elif temp_df['Attack_type'].item() == 'Gradient':
                _, idx = gradients[name+'.weight'].detach().abs().view(-1).topk(k_top)
            elif temp_df['Attack_type'].item() == 'Gradient+Magnitude':
                _, idx  = imp_score.topk(k_top)
 
            for i in idx:
                solution_space.append([name, i.item()])
    return solution_space

def attack_get_loss_acc_new(curated_state_dict, model, sol_set, attack_index=7, task = 'astronomy'):
    solutions = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    for sol in sol_set:
        key, value = sol[0],sol[1]
        if key not in solutions.keys():
            solutions[key] = [value]
        else:  
            solutions[key].append(value)
    # print(solutions)
    for name, param in model.named_parameters():
        if param.dtype==torch.int8 and gradients[name] is not None and name in solutions.keys():
            # print(name, len(solutions[name]))
            if param.dtype==torch.int8:
                w = param.data.detach().contiguous().view(-1)
            else:
                w = param.data.float().detach().view(-1)
            state_dict_copy = copy.deepcopy(curated_state_dict)
            idx = solutions[name]
            
            if param.dtype==torch.int8:
                # print(attack_args['idx'])
                w[idx] = flip_bits_in_tensor(w[idx], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.data.shape)
            else:
                w[idx] = -w[idx]
                state_dict_copy[name+'.weight'] = w.reshape(param.data.shape)
            custom_load_state_dict(model, state_dict_copy)
            clear_memory()
 
    loss, acc = mmlu_loss(model, tokenizer, optimizer, '',[task],mode='zo')[0], mmlu_test(model, tokenizer, '',[task])
    custom_load_state_dict(model, curated_state_dict)
    return loss, acc*100

# Get loss with solution space indices only
def attack_get_loss_new(curated_state_dict, model, sol_set, attack_index=7, task='astronomy'):
    solutions = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    for sol in sol_set:
        key, value = sol[0],sol[1]
        if key not in solutions.keys():
            solutions[key] = [value]
        else:  
            solutions[key].append(value)
    # print(solutions)
    for name, param in model.named_parameters():
        if param.dtype==torch.int8 and name in solutions.keys():
            # print(name, len(solutions[name]))
            if param.dtype==torch.int8:
                w = param.data.detach().contiguous().view(-1)
            else:
                w = param.data.float().detach().view(-1)
            state_dict_copy = copy.deepcopy(curated_state_dict)
            idx = solutions[name]
            
            if param.dtype==torch.int8:
                # print(attack_args['idx'])
                w[idx] = flip_bits_in_tensor(w[idx], attack_index)
                state_dict_copy[name].data[:] = w.reshape(param.data.shape)
            else:
                w[idx] = -w[idx]
                state_dict_copy[name+'.weight'] = w.reshape(param.data.shape)
            custom_load_state_dict(model, state_dict_copy)
            clear_memory()
 
    loss= mmlu_loss(model, tokenizer, optimizer, '',[task],mode='zo')[0]
    custom_load_state_dict(model, curated_state_dict)
    return loss

def sensitivity_ablation(model, tokenizer, optimizer, saved_state_dict, gradients):
    percent_of_weights = [0.00001, 0.0001, 0.001, 0.01, 0.1,1,5,10]
    zo_eps = 1e-3
    layer_sensitivity = {}
    clear_memory()
    custom_load_state_dict(model, saved_state_dict)
    #line 18
    for name, param in model.named_parameters():
        if gradients[name] is not None and 'weight' in name:# and 'language_model.model.embed_tokens' not in name:
            
            attack_args = {'idx' : [0], 'attack_bit' : 3}

            sensitivity = {'Magnitude':[], 'Gradient':[], 'Gradient+Magnitude':[], 'number_of_weights_flipped': [], 'percentage_of_weights_flipped': percent_of_weights}
    
            k_tops =  [int((k/100)*gradients[name].detach().view(-1).size()[0]) for k in percent_of_weights]
            print(name, k_tops)
            if param.dtype==torch.int8:
                w = param.data.detach().contiguous().view(-1)
            else:
                w = param.data.float().detach().view(-1)
            g = gradients[name].float().detach().view(-1)
            # print(w.shape)
    
            imp_score = importance_score(w, g, alpha=0.5)  

            print(f'Layer name: {name}')
    
            for k_top in k_tops:
                # BFLIP
                print(f'k_top: {k_top}')
                wval, w_idx = w.detach().abs().reshape(-1).topk(k_top)
                gval, g_idx = gradients[name].detach().abs().view(-1).topk(k_top)
                ival, i_idx  = imp_score.topk(k_top)
                clear_memory()
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[w_idx] = flip_bits_in_tensor(w[w_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[w_idx] = -w[w_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                l  = mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='zo')[0]
                sensitivity['Magnitude'].append(l)
    
                print(name, "Magnitude based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()
            
                if param.dtype==torch.int8:
                    w = param.data.detach().contiguous().view(-1)
                else:
                    w = param.data.float().detach().view(-1)
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[g_idx] = flip_bits_in_tensor(w[g_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[g_idx] = -w[g_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                l= mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='zo')[0]
    
                sensitivity['Gradient'].append(l)
    
                print(name, "Gradient based:",l)
    
                custom_load_state_dict(model, saved_state_dict)
                clear_memory()
            
                if param.dtype==torch.int8:
                    w = param.data.detach().contiguous().view(-1)
                else:
                    w = param.data.float().detach().view(-1)
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
    
                state_dict_copy = copy.deepcopy(saved_state_dict)
                if param.dtype==torch.int8:
                    w[i_idx] = flip_bits_in_tensor(w[i_idx], 7)
                    # print(attack_args['idx'])
                    state_dict_copy[name].data[:] = w.reshape(state_dict_copy[name].data.shape)
                else:
                    # print(w[w_idx])
                    w[i_idx] = -w[i_idx]
                    state_dict_copy[name] = w.reshape(state_dict_copy[name].data.shape)
                custom_load_state_dict(model, state_dict_copy)
                clear_memory()
    
                l = mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='zo')[0]
            
                sensitivity['Gradient+Magnitude'].append(l)
            
                print(name, "Gradient+magnitude based:",l)
                custom_load_state_dict(model, saved_state_dict)
                sensitivity['number_of_weights_flipped'].append(k_top)
                clear_memory()
            layer_sensitivity[name] = sensitivity
    return layer_sensitivity

# Generate population space with weight values across alpha ablations for min-max importance
def gen_pop_space(model,gradients, percent_of_weights = 0.01, grad_to_mag_ratio=1.0, type=1, random_flag = 0):
    # print("Attacked bit position: ", bit_position_to_flip, ", Percentage of weights attacked: ",percent_of_weights, ", Gradients to magnitude ratio:",grad_to_mag_ratio,',is random?:','yes' if random_flag else 'no'  )
    pop = []
    for name, param in model.named_parameters():
        if 'weight' in name and gradients[name] is not None and param.dtype==torch.int8:
            clear_memory()
        # # custom_load_state_dict(model, curated_state_dict)
        # if 'weight' in name and gradients[name] is not None and param.dtype==torch.int32:
            print(name, param.dtype)
            w1 = param.data
            wf1 = torch.flatten(w1)
            orig_dtype = wf1.dtype
 
            k_top =  int((percent_of_weights/100)*gradients[name].detach().abs().view(-1).size()[0])
            w = param.data.detach().view(-1).to(torch.int8)
            g = gradients[name].detach().abs().view(-1)
 
       
            wval, w_idx = w.topk(k_top)
            gval, g_idx = g.topk(k_top)
            alpha_0_val, alpha_0_idx = importance_score(w, g, alpha=0).topk(k_top)
            alpha_1_val, alpha_1_idx = importance_score(w, g, alpha=1).topk(k_top)
            alpha_025_val, alpha_025_idx = importance_score(w, g, alpha=0.25).topk(k_top)
            alpha_050_val, alpha_050_idx = importance_score(w, g, alpha=0.50).topk(k_top)
            alpha_075_val, alpha_075_idx = importance_score(w, g, alpha=0.75).topk(k_top)
            
            p = []
            for wid, w, gid, g, alpha0id, alpha0, alpha025id, alpha025, alpha050id, alpha050, alpha075id, alpha075, alpha1id, alpha1 in zip(w_idx, wval, g_idx, gval, alpha_0_idx, alpha_0_val, alpha_025_idx, alpha_025_val, alpha_050_idx, alpha_050_val, alpha_075_idx, alpha_075_val, alpha_1_idx, alpha_1_val):
                p.append([name, wid.item(),w.item(), gid.item(), g.item(), alpha0id.item(), alpha0.item(), alpha025id.item(), alpha025.item(), alpha050id.item(), alpha050.item(), alpha075id.item(), alpha075.item(), alpha1id.item(), alpha1.item()])
            pop=pop+p
            clear_memory()
            # custom_load_state_dict(model, saved_state_dict)
            # break
    return pop

# Generate population space with weight values only for some given layers
def gen_pop_space_top_layers(model,gradients, layers, percent_of_weights):
    # print("Attacked bit position: ", bit_position_to_flip, ", Percentage of weights attacked: ",percent_of_weights, ", Gradients to magnitude ratio:",grad_to_mag_ratio,',is random?:','yes' if random_flag else 'no'  )
    pop = []
    for name, param in model.named_parameters():
        if 'weight' in name and gradients[name] is not None and param.dtype==torch.int8 and name in layers:
            clear_memory()
        # custom_load_state_dict(model, curated_state_dict)
        
 
            w1 = param.data
            wf1 = torch.flatten(w1)
 
            k_top =  int((percent_of_weights/100)*gradients[name].detach().abs().view(-1).size()[0])
            w = param.data.detach().view(-1).to(torch.int8)
            g = gradients[name].detach().abs().view(-1)
 
       
            wval, w_idx = w.detach().abs().topk(k_top)
            gval, g_idx = g.topk(k_top)
            i_val,i_idx = importance_score(w, g, alpha=0.5).topk(k_top)
            sum_imp_val, sum_imp_idx = sum_norm_importance_score(w, g).topk(k_top)
            grad_imp_val, grad_imp_idx = grad_weighted_mag_imp_score(w, g).topk(k_top)
            # alpha_025_val, alpha_025_idx = importance_score(w, g, alpha=0.25).topk(k_top)
            # alpha_050_val, alpha_050_idx = importance_score(w, g, alpha=0.50).topk(k_top)
            # alpha_075_val, alpha_075_idx = importance_score(w, g, alpha=0.75).topk(k_top)
            
            p = []
            # for wid, w, gid, g, alpha0id, alpha0, alpha025id, alpha025, alpha050id, alpha050, alpha075id, alpha075, alpha1id, alpha1 in zip(w_idx, wval, g_idx, gval, alpha_0_idx, alpha_0_val, alpha_025_idx, alpha_025_val, alpha_050_idx, alpha_050_val, alpha_075_idx, alpha_075_val, alpha_1_idx, alpha_1_val):
            #     p.append([name, wid.item(),w.item(), gid.item(), g.item(), alpha0id.item(), alpha0.item(), alpha025id.item(), alpha025.item(), alpha050id.item(), alpha050.item(), alpha075id.item(), alpha075.item(), alpha1id.item(), alpha1.item()])
            # pop=pop+p
            for wid, w, gid, g, iid, ival, sumid, sumval, gradid, gradval in zip(w_idx, wval, g_idx, gval, i_idx, i_val, sum_imp_idx, sum_imp_val, grad_imp_idx, grad_imp_val):
                p.append([name, wid.item(),w.item(), gid.item(), g.item(), iid.item(), ival.item(), sumid.item(), sumval.item(), gradid.item(), gradval.item()])
            pop=pop+p
            clear_memory()
            # custom_load_state_dict(model, curated_state_dict)
            # break
    return pop

# Generate population space with weight values only for some given layers with alpha ablations
def gen_pop_space_top_layers_alpha(model,gradients, layers, percent_of_weights):
    # print("Attacked bit position: ", bit_position_to_flip, ", Percentage of weights attacked: ",percent_of_weights, ", Gradients to magnitude ratio:",grad_to_mag_ratio,',is random?:','yes' if random_flag else 'no'  )
    pop = []
    for name, param in model.named_parameters():
        if 'weight' in name and gradients[name] is not None and param.dtype==torch.int8 and name in layers:
            clear_memory()
        # custom_load_state_dict(model, curated_state_dict)
        
 
            w1 = param.data
            wf1 = torch.flatten(w1)
 
            k_top =  int((percent_of_weights/100)*gradients[name].detach().abs().view(-1).size()[0])
            w = param.data.detach().view(-1).to(torch.int8)
            g = gradients[name].detach().abs().view(-1)
 
       
            wval, w_idx = w.detach().abs().topk(k_top)
            gval, g_idx = g.topk(k_top)
            # i_val,i_idx = importance_score(w, g, alpha=0.5).topk(k_top)
            # sum_imp_val, sum_imp_idx = sum_norm_importance_score(w, g).topk(k_top)
            # grad_imp_val, grad_imp_idx = grad_weighted_mag_imp_score(w, g).topk(k_top)
            alpha_025_val, alpha_025_idx = importance_score(w, g, alpha=0.25).topk(k_top)
            alpha_050_val, alpha_050_idx = importance_score(w, g, alpha=0.50).topk(k_top)
            alpha_075_val, alpha_075_idx = importance_score(w, g, alpha=0.75).topk(k_top)
            alpha_0_val, alpha_0_idx = importance_score(w, g, alpha=0).topk(k_top)
            alpha_1_val, alpha_1_idx = importance_score(w, g, alpha=1).topk(k_top)
            
            p = []
            for wid, w, gid, g, alpha0id, alpha0, alpha025id, alpha025, alpha050id, alpha050, alpha075id, alpha075, alpha1id, alpha1 in zip(w_idx, wval, g_idx, gval, alpha_0_idx, alpha_0_val, alpha_025_idx, alpha_025_val, alpha_050_idx, alpha_050_val, alpha_075_idx, alpha_075_val, alpha_1_idx, alpha_1_val):
                p.append([name, wid.item(),w.item(), gid.item(), g.item(), alpha0id.item(), alpha0.item(), alpha025id.item(), alpha025.item(), alpha050id.item(), alpha050.item(), alpha075id.item(), alpha075.item(), alpha1id.item(), alpha1.item()])
            pop=pop+p
            # for wid, w, gid, g, iid, ival, sumid, sumval, gradid, gradval in zip(w_idx, wval, g_idx, gval, i_idx, i_val, sum_imp_idx, sum_imp_val, grad_imp_idx, grad_imp_val):
            #     p.append([name, wid.item(),w.item(), gid.item(), g.item(), iid.item(), ival.item(), sumid.item(), sumval.item(), gradid.item(), gradval.item()])
            # pop=pop+p
            clear_memory()
            # custom_load_state_dict(model, curated_state_dict)
            # break
    return pop

# Get loss and logits after attack on given population space with weights, based on gradient, magnitude, or alpha ablation
def attack_get_loss(model, saved_state_dict, indices, index, attack_index=7):
    solution = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    # model_state_dict = model.state_dict()
    state_dict_copy = copy.deepcopy(model.state_dict())
    for inner_list in indices:
        if isinstance(indices, list):
            key, value = inner_list[0], inner_list[index]
            if key not in solution:
                solution[key] = [value]
            else:
                solution[key].append(value)
    # print(state_dict_copy.keys())
    for name, param in model.named_parameters():
        if 'weight' in name and param.dtype==torch.int8 and name in solution.keys():
            w = param.data.detach().view(-1) #F.dequantize_4bit(param, param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).float().detach().view(-1)
            w[solution[name]]= flip_bits_in_tensor(w[solution[name]], attack_index)
            state_dict_copy[name].data[:] = w.reshape(param.shape)
            custom_load_state_dict_single_layer(model, state_dict_copy, name)
    loss, logits = mmlu_loss(model, tokenizer, optimizer,'', ['astronomy'])
    print(loss)
    for name, param in model.named_parameters():
        if 'weight' in name and name in solution.keys():
            custom_load_state_dict_single_layer(model, saved_state_dict, name)

    for name, param in model.named_parameters():
        if 'weight' in name and name in solution.keys():
            custom_load_state_dict_single_layer(model, saved_state_dict, name)
    # model.load_state_dict(saved_state_dict)
    return loss, logits

# Get accuracy after attack on given population space with weights, based on gradient, magnitude, or alpha ablation
def attack_get_acc(model, saved_state_dict, indices, index, attack_index=7):
    solution = {}
    attack_args = {'idx' : [0], 'attack_bit' : attack_index}
    # model_state_dict = model.state_dict()
    state_dict_copy = copy.deepcopy(model.state_dict())
    for inner_list in indices:
        if isinstance(indices, list):
            key, value = inner_list[0], inner_list[index]
            if key not in solution:
                solution[key] = [value]
            else:
                solution[key].append(value)
    # print(state_dict_copy.keys())
    for name, param in model.named_parameters():
        if 'weight' in name and param.dtype==torch.int8 and name in solution.keys():
            w = param.data.detach().view(-1) #F.dequantize_4bit(param, param.quant_state, quant_type=param.quant_type, blocksize=param.blocksize).float().detach().view(-1)
            w[solution[name]]= flip_bits_in_tensor(w[solution[name]], attack_index)
            state_dict_copy[name].data[:] = w.reshape(param.shape)
            custom_load_state_dict_single_layer(model, state_dict_copy, name)

    acc = mmlu_test(model, tokenizer,'', ['astronomy'])
    print(acc)
    for name, param in model.named_parameters():
        if 'weight' in name and name in solution.keys():
            custom_load_state_dict_single_layer(model, saved_state_dict, name)
    # model.load_state_dict(saved_state_dict)
    return acc

# Set of logarithmically spaced indices
def logarithmic_indices(start, x):
    # Ensure start is at least 0
    if start < 0:
        start = 0
    
    # Initialize indices list starting from the range [start, min(10, x)]
    indices = list(range(start, min(10, x+1)))
    
    power = 1
    while 10 ** power <= x:
        for i in range(1, 10):  # Add multiples of 10^power (10, 20, ..., 90)
            value = i * (10 ** power)
            if value > x:
                break
            if value >= start:
                indices.append(value)
        power += 1
    
    # Include the final value x if it's not already in the list
    if x >= start:
        indices.append(x)
        
    return sorted(set(indices))

def union_mixed_type_lists(list1, list2):

    set1 = {tuple(item) for item in list1}
    set2 = {tuple(item) for item in list2}
    union_set = set1.union(set2)
    result = [list(item) for item in union_set]
    
    return result

def mutate(child, mutation_rate = 0.01):
    mutation_rate = random.randrange(1,10)/100
    temp = []
    for i in range(len(child)):
        if random.random() > mutation_rate:
            temp.append(child[i])
    return temp

def signcompare(l,l_th):
    if l >= l_th:
        return 1
    return -1

def crossover(parent1, parent2, SolutionSpace, crossover_prob=0.9):
    if random.random() < crossover_prob:
        child = []
        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(gene1)
            else:
                child.append(gene2)
    else:
        # child = list(random.sample(SolutionSpace, k=int(num_weights*0.99)))
        child = mutate(parent1)
    return child



def tournament_selection(sol_list):
    mating_pool = []
    while len(mating_pool) < 2:
        i, j = random.sample(range(len(sol_list)), 2)
        # print('Fitnesses:',sol_list[i][0], sol_list[j][0])
        if sol_list[i][0] > sol_list[j][0]:
            # print('returning:',sol_list[i][0])
            mating_pool.append(sol_list[i])
        else:
            # print('returning:',sol_list[j][0])
            mating_pool.append(sol_list[j])
    return mating_pool


def evolutionary_optimization(model, saved_state_dict, pop_space_top1):
    InitNumSol = 50     #initial number of solutions
    max_gen = 150        #Maximum number of iterations/generations
    numSol = 40        #Number of solutions in each generation 
    index = 9
    pop_space_top1.sort(key=lambda x: x[index+1])
    pop_space_top1 = pop_space_top1[::-1]
    SolutionSpace = pop_space_top1[:]
    sol_list = [[0,SolutionSpace,0]]
    loss_progress = []
    loss_threshold  = 7
    clear_memory()

    for i in range(InitNumSol):
        sol_list.append([0, mutate(SolutionSpace),0])

    best_solution = sol_list[0]
    best_loss = 0
    l_progress = [] #For plotting
    opt_progress = []
    progress = {} #For plotting
    iterations_data = [] # snaping data from intermediate iterations
    interStep = 5 # iteration steps after snapping data

    custom_load_state_dict(model, saved_state_dict)
    while(max_gen>0):
        clear_memory()
        max_gen = max_gen - 1
        # Calculate loss per solution
        for i in range(len(sol_list)):
            loss = attack_get_loss(model, saved_state_dict, sol_list[i][1], index)[0]
            f = signcompare(loss, loss_threshold)*(loss/len(sol_list[i][1]))
            print(loss, len(sol_list[i][1]), f)
            sol_list[i][0] = f
            sol_list[i][2] = loss
            
        # rank the solutions based on their loss
        sol_list.sort(reverse = True)

            
        # collect progress information
        progress[-max_gen] = sol_list[0][2]

        if loss > best_loss:
            best_loss = loss
            best_solution = sol_list[0]
        best_solution = sol_list[0]
        # for i in range(len(sol_list)):
        #     if sol_list[i][0] > loss_threshold and len(sol_list[i][1]) < len(best_solution[1]):
        #         best_solution = sol_list[i]
        l_progress.append(len(best_solution[1]))
        opt_progress.append(best_solution[1])
        loss_progress.append(best_solution[2])
        loss = attack_get_loss(model, saved_state_dict, best_solution[1], index)[0]
        print('-----------------Generation:',max_gen, 'Loss:', best_solution[0], 'Solution length:', len(best_solution[1]))


        sol_list2 = [best_solution,[0,mutate(best_solution[1], mutation_rate = 0.01), 0]] 


        for j in range(numSol-len(sol_list2)):
            #line 17 Algorithm 3 
            a1 = tournament_selection(sol_list)
            parent1 = a1[0][1]
            parent2 = a1[1][1]

            if random.random() > 0.5:
                child = crossover(best_solution[1], parent1, SolutionSpace, crossover_prob=0.9)
                child = [0,mutate(child, mutation_rate = 0.01), 0]
            else:
                child = crossover(best_solution[1], parent2, SolutionSpace, crossover_prob=0.9)
                
                child = [0,mutate(child, mutation_rate = 0.01), 0]
            
            sol_list2.append(child)

        # Use this solution list for next iteration
        sol_list = sol_list2

    attnbreaker_gen_df = pd.DataFrame({'Num_Generations':range(len(l_progress)), 'Num_Weights':l_progress, 'Loss':loss_progress})
    attnbreaker_gen_df.to_csv('./log_results/llama_3_2_1b_instruct_int8_new_genetic_sweep_v2.csv')
    return best_solution[1], attnbreaker_gen_df

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    notebook_login()

    clear_memory()

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model, tokenizer = load_model(model_name, mode = 'int8')

    print(f'Memory footprint of the model: {model.get_memory_footprint()}')

    wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    perplexity = calculate_perplexity(model, tokenizer, wiki_data, size = 10)
    print(f'Perplexity: {perplexity}')

    # Optimizer
    optimizer = Adam8bit(model.parameters(), lr=2e-5)

    # Calculate MMLU Accuracy, loss and logits for test
    mmlu_test(model, tokenizer,'',['astronomy']) 
    loss, logits = mmlu_loss(model,tokenizer,optimizer,'',['astronomy'])
    print(f'Model loss: {loss}')

    # Disable quantization flag
    model.is_quantized = False

    # Cast quantized weights to float16
    for name, param in model.named_parameters():
        if param.dtype == torch.int8:
            param.data = param.data.to(torch.float16)
            param.requires_grad = True

    # Calculate loss with gradients
    mmlu_loss(model, tokenizer, optimizer, '',['astronomy'],mode='bp')

    # Store gradients
    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = param.grad

    # Clear cache and get original state dictionary
    clear_memory()
    saved_state_dict = copy.deepcopy(model.state_dict())

    # Clear Cache
    clear_memory()

    # Reload quantized model
    model, tokenizer = load_model(model_name, mode = 'int8')

    # Clear cache
    clear_memory()

    # Sensitivity analysis
    print(f'Sensitivity Analysis.......')
    layer_sensitivity = sensitivity_study(model, tokenizer, optimizer, saved_state_dict, gradients)
    clear_memory()

    # Sanity Check on model integrity
    print(f'Sanity check after sensitivity Analysis.......')
    custom_load_state_dict(model, saved_state_dict)
    perplexity, loss = calculate_perplexity(model,tokenizer,wiki_data, size=20), mmlu_loss(model, tokenizer, optimizer, '', ['astronomy'])[0]
    print(f'WikiText perplexity: {perplexity}')
    print(f'MMLU Loss: {loss}')

    # Plot layer sensitivity

    fig = plt.figure(1)
    
    x =range(len(layer_sensitivity))
    
    y1, y2, y3 = [],[], []
    for i in layer_sensitivity:
        y1.append(i[1])
        y2.append(i[3])
        y3.append(i[5])
    
    plt.scatter(x, y1, c ="blue", label='Magnitude based attack')
    plt.scatter(x, y2, c ="red", label='Gradient based attack')
    # plt.scatter(x, y3, c ="green", label='Magnitude+Gradient based attack')
    plt.xlabel("Layer Id")
    plt.ylabel("Model loss")
    # To show the plot
    plt.legend()
    plt.grid()
    plt.show()

    df = pd.DataFrame()
    df['Layer name'] = []
    df['Magnitude attack loss'] = []
    df['Magnitude attack perplexity'] = []
    df['gradient attack loss'] = []
    df['gradient attack perplexity'] = []
    df['magnitude_+_Gradient attack loss'] = []
    df['magnitude_+_Gradient attack perplexity'] = []
    for i in layer_sensitivity:
        df.loc[len(df.index)] = [i[0], i[1], i[2], i[3], i[4],i[5],i[6]]

    print(df)

    df.to_csv('./log_results/llama_3_2_1b_instruct_int8_fp16grad_sensitivity_5%_sign_bit_flip_v2.csv')
    k=20
    df2 = df.sort_values(by=['Magnitude attack loss'], ascending=False)[:k]
    df1 = df.sort_values(by=['gradient attack loss'], ascending=False)[:k]
    df3 = df.sort_values(by=['magnitude_+_Gradient attack loss'], ascending=False)[:k]
    df4 = pd.merge(df1, df2, how ='inner')
    top=df3[df3.columns[0]][:5].tolist()
    print(top)

    print(f'Sensitivity Ablation.......')
    layer_sensitivity = sensitivity_ablation(model, tokenizer, optimizer, saved_state_dict, gradients)
    clear_memory()

    # Store ablations
    flattened_data = []
    for layer, sensitivities in layer_sensitivity.items():
        percent_of_weights_flipped = sensitivities['percentage_of_weights_flipped']
        number_of_weights_flipped = sensitivities['number_of_weights_flipped']
        for sensitivity_type, values in sensitivities.items():
            if sensitivity_type not in ['percentage_of_weights_flipped', 'number_of_weights_flipped']:
                for index, value in enumerate(values):
                    flattened_data.append({
                        'Layer': layer,
                        'Attack_type': sensitivity_type,
                        'percentage_of_weights_flipped': percent_of_weights_flipped[index],
                        'number_of_weights_flipped': number_of_weights_flipped[index],
                        'Model_loss': value
                    })
    # Create a DataFrame
    df = pd.DataFrame(flattened_data)
    # Save to CSV
    df.to_csv('./log_results/llama_3_2_1b_instruct_int8_layers_sensitivity_full_sign_bit_flip_v2.csv', index=False)
    print(df)

    df = pd.read_csv('./log_results/llama_3_2_1b_instruct_int8_layers_sensitivity_full_sign_bit_flip_v2.csv')
    # df = df[df['percentage_of_weights_flipped'] == 5]
    df1 = df[df['Attack_type']=='Gradient']
    df2 = df[df['Attack_type']=='Magnitude']
    df3 = df[df['Attack_type']=='Gradient+Magnitude']

    dtypes = []
    for i in range(df3.shape[0]):
        if model.state_dict()[df3['Layer'].iloc[i]].dtype == torch.int8:
            dtypes.append('int8')
        else:
            dtypes.append('float16')

    df3['dtype'] = dtypes
    df3_n = df3[df3['dtype'] == 'int8']

    plt.figure(figsize=(8, 6))
    marker = itertools.cycle((',', '+', '.', 'o', '*', 'x', 'h','^')) 
    plt.rcParams.update({'font.size':14})

    # plt.scatter(list(range(1, df1.shape[0]+1)), df1['Model_loss'], color='b', label='Absolute Gradient', alpha=0.5)
    # plt.scatter(list(range(1, df2.shape[0]+1)), df2['Model_loss'], color='g', label='Absolute Magnitude', alpha=0.5)
    plt.scatter(list(range(1, df3_n.shape[0]+1)), df3_n['Model_loss'], color='r', label='Hybrid Sensitivity Score')
    plt.ylim([1.3, 10])
    plt.grid()
    plt.legend()
    plt.xlabel('Layer ID')
    plt.ylabel('Model Loss')
    plt.savefig('./log_results/llama_3_2_1b_instruct_int8_layer_sens_5_v2.pdf')
    plt.show()

    int_8_layers = []
    for name, param in model.named_parameters():
        if param.dtype==torch.int8 and gradients[name] is not None:
            int_8_layers.append(name)

    # Get most sensitive layer
    df_int_8 = df3[df3['Layer'].isin(int_8_layers)]
    top_5_layers = df_int_8[df_int_8['Model_loss']>=4]
    sorted_top_5_layers = top_5_layers.sort_values(by=['Model_loss'],ascending=False)
    sorted_top_5_layers = sorted_top_5_layers.sort_values(by=['number_of_weights_flipped'],ascending=True)
    sorted_top_5_layers = sorted_top_5_layers.drop_duplicates(subset='Layer', keep='first')

    df_int_8 = df[df['Layer'].isin(int_8_layers)]
    all_layers = df_int_8[df_int_8['Model_loss']>=7]
    sorted_layers = all_layers.sort_values(by=['Model_loss'],ascending=False)
    sorted_layers = sorted_layers.sort_values(by=['number_of_weights_flipped'],ascending=True)
    sorted_layers = sorted_layers.drop_duplicates(subset='Layer', keep='first')
    sorted_layers['percentage_of_weights_flipped'] = sorted_layers['percentage_of_weights_flipped'].iloc[0]

    # Sanity Check on model integrity
    print(f'Sanity check after sensitivity ablation.......')
    custom_load_state_dict(model, saved_state_dict)
    perplexity, loss = calculate_perplexity(model,tokenizer,wiki_data, size=20), mmlu_loss(model, tokenizer, optimizer, '', ['astronomy'])[0]
    print(f'WikiText perplexity: {perplexity}')
    print(f'MMLU Loss: {loss}')

    # Population space with weights for most sensitive layer
    print(f'Getting population space with weights for most sensitive layer......')
    pop_space_top1 = gen_pop_space_top_layers(model, gradients, [sorted_top_5_layers['Layer'].iloc[0]], 2)

    # Population space with weights for most sensitive layer with alpha ablations
    print(f'Getting population space with weights for most sensitive layer with alpha ablations......')
    pop_space_top1 = gen_pop_space_top_layers_alpha(model, gradients, [sorted_top_5_layers['Layer'].iloc[0]], 2)

    # Get population spaces
    print(f'Getting population spaces......')
    sol_space_any = gen_pop_space_top_layers(model, gradients, list(sorted_layers['Layer']), 2) # all layers
    print(f'Weights from all layers: {len(sol_space_any)}')
    sol_space_1 = gen_pop_space_top_layers(model, gradients, list(sorted_top_5_layers['Layer'].iloc[0:1]), 2) # most sensitive
    print(f'Weights from most sensitive layer: {len(sol_space_1)}')
    sol_space_3 = gen_pop_space_top_layers(model, gradients, list(sorted_top_5_layers['Layer'].iloc[0:3]), 2) # top3
    print(f'Weights from top 3 most sensitive layer: {len(sol_space_3)}')
    sol_space_5 = gen_pop_space_top_layers(model, gradients, list(sorted_top_5_layers['Layer'].iloc[0:5]), 2) # top5
    print(f'Weights from top 5 most sensitive layer: {len(sol_space_5)}')

    # Normalization method on most sensitive layer
    print(f'Normalization method ablation on most sensitive layer........')
    clear_memory()

    loss_mag = []
    loss_grad = []
    loss_imp = []
    loss_sum_imp = []
    loss_grad_imp = []
    custom_load_state_dict(model, saved_state_dict)

    num_weights = logarithmic_indices(1000, len(sol_space_1))
    print(num_weights)

    for n in num_weights:
        print(f'num weights = {n}')
        clear_memory()
        sol_space_1.sort(key=lambda x: x[2])
        sol_space_1 = sol_space_1[::-1]
        loss_mag.append(attack_get_loss(model, saved_state_dict, sol_space_1[:n], 1, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[4])
        sol_space_1 = sol_space_1[::-1]
        loss_grad.append(attack_get_loss(model, saved_state_dict, sol_space_1[:n], 3, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[6])
        sol_space_1 = sol_space_1[::-1]
        loss_imp.append(attack_get_loss(model, saved_state_dict, sol_space_1[:n], 5, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[8])
        sol_space_1 = sol_space_1[::-1]
        loss_sum_imp.append(attack_get_loss(model, saved_state_dict, sol_space_1[:n], 7, 7)[0])

        clear_memory()
        sol_space_1.sort(key=lambda x: x[10])
        sol_space_1 = sol_space_1[::-1]
        loss_grad_imp.append(attack_get_loss(model, saved_state_dict, sol_space_1[:n], 9, 7)[0])

    loss_df = pd.DataFrame({'Num_Weights':num_weights, 'Magnitude': loss_mag, 'Gradient': loss_grad, 'Min_Max_Importance': loss_imp, 'Sum_Importance': loss_sum_imp, 'Grad_Importance':loss_grad_imp})
    loss_df.to_csv('./log_results/llama_3_2_1b_instruct_int8_importance_var_v3.csv', index=False)
    print(loss_df)

    marker = itertools.cycle(['.', '+', '*', '^'])
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size':14})

    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Magnitude'], marker = next(marker), label = 'Absolute Magnitude')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Gradient'], marker = next(marker), label = 'Absolute Gradient')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Min_Max_Importance'], marker = next(marker), label = 'Min-Max Normalization')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Grad_Importance'], marker = next(marker), label = 'Sum Normalization')
    plt.plot(loss_df['Num_Weights']*100/model.get_memory_footprint()/8, loss_df['Sum_Importance'], marker = next(marker), label = 'Gradient-Weighted \nMagnitude Score')
    plt.xlabel('Fault Rate (in %)')
    plt.ylabel('Model Loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('./log_results/llama_3_2_1b_instruct_int8_importance_v3.pdf')
    plt.show()

    # Top layer sensitivity
    print(f'Check layer sensitivity........')
    loss_val_any = []
    loss_val_1 = []
    loss_val_3 = []
    loss_val_5 = []

    custom_load_state_dict(model, saved_state_dict)

    num_weights = [0]+logarithmic_indices(int(1e3), int(1e6))
    print(num_weights)
    index = 5

    for n in num_weights:
        print(f'num weights = {n}')

        sol_space_any.sort(key=lambda x: x[index+1])
        sol_space_any = sol_space_any[::-1]
        loss_val_any.append(attack_get_loss(model, saved_state_dict, sol_space_any[:n], index)[0])

        sol_space_1.sort(key=lambda x: x[index+1])
        sol_space_1 = sol_space_1[::-1]
        loss_val_1.append(attack_get_loss(model, saved_state_dict, sol_space_1[:n], index)[0])

        sol_space_3.sort(key=lambda x: x[index+1])
        sol_space_3 = sol_space_3[::-1]
        loss_val_3.append(attack_get_loss(model, saved_state_dict, sol_space_3[:n], index)[0])

        sol_space_5.sort(key=lambda x: x[index+1])
        sol_space_5 = sol_space_5[::-1]
        loss_val_5.append(attack_get_loss(model, saved_state_dict, sol_space_5[:n], index)[0])

    sol_space_loss_df = pd.DataFrame({'Num_Weights': num_weights, 'Sol_Space_Any': loss_val_any, 'Sol_Space_1': loss_val_1, 'Sol_Space_3': loss_val_3, 'Sol_Space_5': loss_val_5})
    sol_space_loss_df.to_csv('./log_results/llama_3_2_1b_instruct_int8_sol_space_losses_v2.csv')
    sol_space_loss_df = pd.read_csv('./log_results/llama_3_2_1b_instruct_int8_sol_space_losses_v2.csv').iloc[:, 1:]
    print(sol_space_loss_df)

    marker = itertools.cycle(['.', '+', '*', '^'])
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size':14})

    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_Any'], marker = next(marker), label = 'All Layers')
    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_1'], marker = next(marker), label = 'Top-1 Layer')
    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_3'], marker = next(marker), label = 'Top-3 Layers')
    plt.plot(sol_space_loss_df['Num_Weights']*100/model.get_memory_footprint()/8, sol_space_loss_df['Sol_Space_5'], marker = next(marker), label = 'Top-5 Layers')
    plt.grid()
    plt.xlabel('Fault Rate (in %)')
    plt.ylabel('Model Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./log_results/llama_3_2_1b_instruct_int8_topn_layers_v3.pdf')
    plt.show()

    # alpha sweep
    print(f'Ablation study on the alpha values on the most sensitive layer using min-max normalization')
    loss_val_mag = []
    loss_val_grad = []
    loss_val_alpha0 = []
    loss_val_alpha025 = []
    loss_val_alpha05 = []
    loss_val_alpha075 = []
    loss_val_alpha1 = []

    pop_space = pop_space_top1

    for i in [0]+logarithmic_indices(int(1e2), len(pop_space))[:-1]:
        print(f'num weights = {i}')
        pop_space.sort(key=lambda x: x[2])
        pop_space = pop_space[::-1]
        loss_val_mag.append(attack_get_loss(model, saved_state_dict, pop_space[:i], 1, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[4])
        pop_space = pop_space[::-1]
        loss_val_grad.append(attack_get_loss(model, saved_state_dict, pop_space[:i], 3, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[6])
        pop_space = pop_space[::-1]
        loss_val_alpha0.append(attack_get_loss(model, saved_state_dict, pop_space[:i], 5, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[8])
        pop_space = pop_space[::-1]
        loss_val_alpha025.append(attack_get_loss(model, saved_state_dict, pop_space[:i], 7, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[10])
        pop_space = pop_space[::-1]
        loss_val_alpha05.append(attack_get_loss(model, saved_state_dict, pop_space[:i], 9, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[12])
        pop_space = pop_space[::-1]
        loss_val_alpha075.append(attack_get_loss(model, saved_state_dict, pop_space[:i], 11, attack_index=7)[0])

        pop_space.sort(key=lambda x: x[14])
        pop_space = pop_space[::-1]
        loss_val_alpha1.append(attack_get_loss(model, saved_state_dict, pop_space[:i], 13, attack_index=7)[0])

    mag_grad_loss_any = pd.DataFrame({'num_weights' : [0]+logarithmic_indices(int(1e2), len(pop_space))[:-1], 
                                    'mag_loss' : loss_val_mag, 
                                    'grad_loss' : loss_val_grad,
                                    'alpha0_loss' : loss_val_alpha0,
                                    'alpha025_loss' : loss_val_alpha025,
                                    'alpha05_loss' : loss_val_alpha05,
                                    'alpha075_loss' : loss_val_alpha075,
                                    'alpha1_loss' : loss_val_alpha1,
                                    })
    mag_grad_loss_any.to_csv('./log_results/llama_3_2_1b_instruct_int8_fp16grad_popspacetop1_alpha_sweep_v2.csv')

    marker = itertools.cycle((',', '+', '.', 'o', '*', 'x', 'h','^')) 
    pop_mag_df = pd.read_csv('./log_results/llama_3_2_1b_instruct_int8_fp16grad_popspacetop1_alpha_sweep_v2.csv')
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size':14})
    # plt.semilogx(pop_mag_df['num_weights'], pop_mag_df['mag_loss'], label = 'Magnitude-based attack', marker = next(marker))
    # plt.semilogx(pop_mag_df['num_weights'], pop_mag_df['grad_loss'], label = 'Gradient-based attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha0_loss'], label = 'Alpha=0 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha025_loss'], label = 'Alpha=0.25 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha05_loss'], label = 'Alpha=0.5 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha075_loss'], label = 'Alpha=0.75 attack', marker = next(marker))
    plt.plot(pop_mag_df['num_weights']*100/model.get_memory_footprint()/8, pop_mag_df['alpha1_loss'], label = 'Alpha=1 attack', marker = next(marker))
    plt.xlabel('Fault Rate (in %)')
    plt.ylabel('Model Loss')
    plt.grid()
    plt.legend()
    plt.savefig('./log_results/llama_3_2_1b_instruct_int8_alpha_sweep_v3.pdf')
    plt.show()

    print(f'Performing evolutionary optimization......')
    optimized_solution, gen_df = evolutionary_optimization(model, saved_state_dict, pop_space_top1)
    
    plt.rcParams.update({'font.size':14})
    fig, ax1 = plt.subplots(figsize=(8,6))

    ax1.plot(range(gen_df['Num_Generations'].shape[0]), gen_df['Num_Weights'], marker = '+', color='navy', label = 'Solution Length Progression')
    ax1.set_xlabel('Number of Generations')
    ax1.set_ylabel('Number of Weights')
    ax1.legend(loc = 'upper right', bbox_to_anchor=(1,0.9725))

    ax2 = ax1.twinx()
    ax2.plot(range(gen_df['Num_Generations'].shape[0]), gen_df['Loss'], marker = '+', color='tomato', label = 'Loss Progression')
    ax2.set_xlabel('Number of Generations')
    ax2.set_ylabel('Model Loss')
    ax2.legend(loc = 'upper right', bbox_to_anchor=(1,0.9))

    plt.grid()
    plt.tight_layout()
    plt.savefig('./log_results/llama_3_2_1b_instruct_int8_new_genetic_sweep_v2.pdf')
    plt.show()

    file_path = './log_results/llama_3_2_1b_instruct_int8_critical_weights.json'
    with open(file_path, 'w') as file:
        json.dump(optimized_solution, file, indent=4)
    print(f'data saved in {file_path}')

    del model, saved_state_dict, tokenizer, gradients

    print('Hello World')