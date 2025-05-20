import torch
from torch import nn
from transformers import Trainer
import json
from datasets import load_metric,Dataset,DatasetDict,load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, Blip2Processor
from transformers import AutoTokenizer
from torch.autograd import grad
import os
import torch
import pandas as pd
from rouge_score import rouge_scorer
import sys
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from typing import List, Optional, Tuple, Union
from typing import Optional
sys.path.insert(0, os.getcwd()+'/custom_transform')
from modeling_blip_2 import Blip2Model,Blip2ForConditionalGeneration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from peft import get_peft_model, PeftConfig
import re
from sklearn.metrics import f1_score

import json, random, time, os, base64
import numpy as np
from pprint import pprint
from collections import Counter, defaultdict
# import cv2
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import torch.nn.functional as F


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
model_NER = "dslim/bert-base-NER"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_NER)
model_NER = AutoModelForTokenClassification.from_pretrained(model_NER).to(device)

# Setup NER pipeline
nlp = pipeline("ner", model=model_NER, tokenizer=tokenizer, device=0 if device == 'cuda' else -1)
print("custom_loss_si_RL")

def clean_decoded_text(text):
    text = text.replace("system\n", "").replace("user\n", "").replace("assistant\n", "").strip()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove gibberish
    return text

# Function to extract named entities
def extract_entities(text):
    ner_results = nlp(text)
    # print(ner_results)
    entities = set()
    for entity in ner_results:
        # if entity['entity'].startswith("B"):  # Start of a new entity
        #     entities.add((entity['word'], entity['entity'][2:]))
        entities.add((entity['word'], entity['entity'][2:]))
    return entities


def weighted_average(values, weights=None):
    if weights is None:
        weights = [1] * len(values)  # Default equal weights if not provided
    
    if len(values) != len(weights):
        raise ValueError("Number of values and weights must be the same")
    
    weighted_sum = sum(value * weight for value, weight in zip(values, weights))
    total_weight = sum(weights)
    
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    
    return weighted_sum / total_weight


g_old_log_probs1 = []
g_old_log_probs2 = []
g_old_log_probs3 = []

# m_id = "microsoft/Phi-3.5-vision-instruct" 
m_id = "llava-hf/llava-1.5-7b-hf"
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
qwenprocessor = AutoProcessor.from_pretrained(m_id,trust_remote_code=True)
from trl import SFTTrainer
# class CustomTrainer(Seq2SeqTrainer):
class CustomTrainer(SFTTrainer):
    def __init__(self, *args, clip_range=0.2, gamma=0.99, gae_lambda=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_range = clip_range  # PPO clipping range
        self.gamma = gamma            # Discount factor
        self.gae_lambda = gae_lambda  # GAE lambda for advantage estimation
        
    
    def compute_rouge_reward(self, preds, labels):
        """Compute the ROUGE reward between predictions and gold labels."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rewards = []
        for i in range(len(preds)):
            score = scorer.score(labels[i], preds[i])
            # reward = (score['rouge1'].fmeasure + score['rougeL'].fmeasure) / 2.0
            reward = (score['rougeL'].fmeasure)
            rewards.append(reward)
        return torch.tensor(rewards).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
        # Extract labels
        global g_old_log_probs1  # Store this during sampling
        global g_old_log_probs2 
        global g_old_log_probs3 
        tokenizer = qwenprocessor
        normalise_tanh = nn.Tanh()
        labels1 = inputs.pop("labels")

        labels2 = inputs.pop("gpt4v")
        labels3 = inputs.pop("gemini")
        label_rag = inputs.pop("rag")
        # print(" customloss inputs",inputs)

        # Forward pass to get logits for three models
        inputs['labels']=labels1
        
        # Phi 3.5 V
        # inputs1 = {'input_ids':inputs["vanilla_inputs_input_ids"],'pixel_values':inputs["pixel_values"],
        #            'image_sizes':inputs["image_sizes"], 'labels':inputs["labels"]}
        
        # # LLAVA
        inputs1 = {'input_ids':inputs["vanilla_inputs_input_ids"],'pixel_values':inputs["pixel_values"],
                   'attention_mask':inputs["vanilla_inputs_attention_mask"], 'labels':inputs["labels"]}
        

        # print("inputs1 Backend: ",inputs1)
        l1, logits1, value_1 = model(**inputs1, return_dict=False)

        inputs['labels'] = labels2
        
        # Phi 3.5 V

        # inputs2 = {'input_ids':inputs["gpt4_inputs_input_ids"],'pixel_values':inputs["pixel_values"],
        #            'image_sizes':inputs["image_sizes"], 'labels':inputs["labels"]}
        
        # LLAVA
        inputs2 = {'input_ids':inputs["gpt4_inputs_input_ids"],'pixel_values':inputs["pixel_values"],
                   'attention_mask':inputs["gpt4_inputs_attention_mask"], 'labels':inputs["labels"]}

        
        l2, logits2, value_2 = model(**inputs2, return_dict=False)

        inputs['labels'] = labels3
        # Phi 3.5 V
        # inputs3 = {'input_ids':inputs["gemini_inputs_input_ids"],'pixel_values':inputs["pixel_values"],
        #            'image_sizes':inputs["image_sizes"], 'labels':inputs["labels"]}

        # # LLAVA
        inputs3 = {'input_ids':inputs["gemini_inputs_input_ids"],'pixel_values':inputs["pixel_values"],
                   'attention_mask':inputs["gemini_inputs_attention_mask"], 'labels':inputs["labels"]}

        l3, logits3, value_3 = model(**inputs3,  return_dict=False)

        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        probs3 = F.softmax(logits3, dim=-1)

        # print("probs1.shape",probs1.shape)
        # Decoding the predicted tokens and gold labels
        
        preds1 = probs1.argmax(dim=-1)
        preds2 = probs2.argmax(dim=-1)
        preds3 = probs3.argmax(dim=-1)

        decoded_preds1 = tokenizer.batch_decode(preds1, skip_special_tokens=True) #y
        decoded_preds2 = tokenizer.batch_decode(preds2, skip_special_tokens=True)
        decoded_preds3 = tokenizer.batch_decode(preds3, skip_special_tokens=True)
        # print("Backend decoded_preds3:",decoded_preds3)

        labels1_temp= labels1.view(labels1.size(0), -1)
        # Now labels1 is [2, 121*1024] = [2, 123904]

        # 2) Convert to Python lists of lists of int:
        labels1_list = labels1_temp.tolist()

        labels2_temp= labels2.view(labels2.size(0), -1)
        labels2_list = labels2_temp.tolist()
        labels3_temp= labels3.view(labels3.size(0), -1)
        labels3_list = labels3_temp.tolist()

        decoded_labels1 = tokenizer.batch_decode([[token_id for token_id in label if token_id != -100] for label in labels1], skip_special_tokens=True)#y_preds1
        decoded_labels2 = tokenizer.batch_decode([[token_id for token_id in label if token_id != -100] for label in labels2], skip_special_tokens=True)
        decoded_labels3 = tokenizer.batch_decode([[token_id for token_id in label if token_id != -100] for label in labels3], skip_special_tokens=True)
        
        label_rag_temp= labels3.view(label_rag.size(0), -1)
        label_rag_list = label_rag_temp.tolist()
        # rag = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in label_rag_list]
        rag = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in label_rag]

        # ROUGE-based rewards
        rewards1 = self.compute_rouge_reward(decoded_preds1, decoded_labels1)
        rewards2 = self.compute_rouge_reward(decoded_preds2, decoded_labels2)
        rewards3 = self.compute_rouge_reward(decoded_preds3, decoded_labels3)

        # Baseline (value approximation) and advantage calculation
        # values = (rewards1 + rewards2 + rewards3) / 3.0


        # print(f"value_1: {value_1}, value_2: {value_2}, value_3: {value_3}")
        values = (value_1 + value_2 + value_3) / 3.0
        
        advantages = rewards1 + rewards2 + rewards3 - values
        index_for_gather = labels1
        index_2_for_gather = labels2
        index_3_for_gather = labels3
        # print("advantages",advantages)
        # try: 
        if len(g_old_log_probs1)!=0 and len(g_old_log_probs1)!=0 and len(g_old_log_probs1)!=0:
            # print("chck")
              # Retrieve old log_probs from stored buffer (inputs should contain them or from some buffer mechanism)
            old_log_probs1 = g_old_log_probs1 # Store this during sampling
            old_log_probs2 = g_old_log_probs2
            old_log_probs3 = g_old_log_probs3
            # print("old_log_probs3:",old_log_probs3)
        # except:
        else:
            # print("nope")
            
            with torch.no_grad():
            # Compute old log probabilities (for PPO clipping)
                try:
                    labels1 = torch.clamp(labels1, 0, probs1.shape[-1] - 1)
                    labels2 = torch.clamp(labels2, 0, probs2.shape[-1] - 1)
                    labels3 = torch.clamp(labels3, 0, probs3.shape[-1] - 1)

                    old_log_probs1 = torch.log(probs1.gather(-1, labels1.detach().unsqueeze(-1)).squeeze(-1))
                    old_log_probs2 = torch.log(probs2.gather(-1, labels2.detach().unsqueeze(-1)).squeeze(-1))
                    old_log_probs3 = torch.log(probs3.gather(-1, labels3.detach().unsqueeze(-1)).squeeze(-1))
                except:
                    print("Except block Compute old log probabilities (for PPO clipping)")
                    # Flatten the last two dims => shape [2, 121*1024] = [2, 123904]
                    labels1_gather = labels1.view(2, -1)

                    # If you only want the first 1360 tokens:
                    labels1_gather = labels1_gather[:, :probs1.size()[1]]  # Now shape is [2, 1360]

                    # Then gather works:
                    index_for_gather = labels1_gather.unsqueeze(-1) 

                    labels2_gather = labels2.view(2, -1)
                    labels2_gather = labels2_gather[:, :probs2.size()[1]]  # Now shape is [2, 1360]
                    index_2_for_gather = labels2_gather.unsqueeze(-1) 

                    labels3_gather = labels3.view(2, -1)
                    labels3_gather = labels3_gather[:, :probs3.size()[1]]  # Now shape is [2, 1360]
                    index_3_for_gather = labels3_gather.unsqueeze(-1) 

                    old_log_probs1 = torch.log(probs1.gather(-1,  index_for_gather).squeeze(-1))
                    old_log_probs2 = torch.log(probs2.gather(-1, index_2_for_gather).squeeze(-1))
                    old_log_probs3 = torch.log(probs3.gather(-1, index_3_for_gather).squeeze(-1))
        
        try:
            # Forward pass again for current log probabilities

            labels1 = torch.clamp(labels1, 0, logits1.shape[-1] - 1)
            labels2 = torch.clamp(labels2, 0, logits2.shape[-1] - 1)
            labels3 = torch.clamp(labels3, 0, logits3.shape[-1] - 1)

            log_probs1 = F.log_softmax(logits1, dim=-1).gather(-1, labels1.unsqueeze(-1)).squeeze(-1)
            log_probs2 = F.log_softmax(logits2, dim=-1).gather(-1, labels2.unsqueeze(-1)).squeeze(-1)
            log_probs3 = F.log_softmax(logits3, dim=-1).gather(-1, labels3.unsqueeze(-1)).squeeze(-1)
        except:
             # Forward pass again for current log probabilities
            print("Except block Forward pass again for current log probabilities")
            log_probs1 = F.log_softmax(logits1, dim=-1).gather(-1, index_for_gather).squeeze(-1)
            log_probs2 = F.log_softmax(logits2, dim=-1).gather(-1, index_2_for_gather).squeeze(-1)
            log_probs3 = F.log_softmax(logits3, dim=-1).gather(-1, index_3_for_gather).squeeze(-1)
        
        g_old_log_probs1 = log_probs1.detach()  # Store this during sampling
        g_old_log_probs2 = log_probs2.detach()
        g_old_log_probs3 = log_probs3.detach()

        if log_probs1.shape[-1] < old_log_probs1.shape[-1]:
            old_log_probs1 = old_log_probs1[:,:log_probs1.shape[-1]]  # Truncate to match the size of log_probs1
        else: 
            # log_probs1 = log_probs1[:,:old_log_probs1.shape[-1]]  # Truncate to match the size of old_log_probs1
            # Calculate how much padding is needed
            pad_size = log_probs1.shape[-1] - old_log_probs1.shape[-1]
            # Pad old_log_probs1 on the right (end of the last dimension)
            old_log_probs1 = torch.nn.functional.pad(old_log_probs1, (0, pad_size), 'constant', 0)

        if log_probs2.shape[-1] < old_log_probs2.shape[-1]:
            old_log_probs2 = old_log_probs2[:,:log_probs2.shape[-1]]  # Truncate to match the size of log_probs2
        else: 
            # log_probs2 = log_probs2[:,:old_log_probs2.shape[-1]]  # Truncate to match the size of old_log_probs2

            # Calculate how much padding is needed
            pad_size = log_probs2.shape[-1] - old_log_probs2.shape[-1]
            # Pad old_log_probs1 on the right (end of the last dimension)
            old_log_probs2 = torch.nn.functional.pad(old_log_probs2, (0, pad_size), 'constant', 0)

        if log_probs3.shape[-1] < old_log_probs3.shape[-1]:
            old_log_probs3 = old_log_probs3[:,:log_probs3.shape[-1]]  # Truncate to match the size of log_probs3
        else: 
            # log_probs3 = log_probs3[:,:old_log_probs3.shape[-1]]  # Truncate to match the size of old_log_probs3

            # Calculate how much padding is needed
            pad_size = log_probs3.shape[-1] - old_log_probs3.shape[-1]
            # Pad old_log_probs1 on the right (end of the last dimension)
            old_log_probs3 = torch.nn.functional.pad(old_log_probs3, (0, pad_size), 'constant', 0)


        try:
            # Calculate the PPO loss using the surrogate objective
            ratios1 = torch.exp(log_probs1 - old_log_probs1)
            ratios2 = torch.exp(log_probs2 - old_log_probs2)
            ratios3 = torch.exp(log_probs3 - old_log_probs3)
        except:
            print("Except block Expand old_log_probs1 to match log_probs1's shape")
 
            log_probs1_reduced = log_probs1.mean(dim=1)
            log_probs2_reduced = log_probs2.mean(dim=1)
            log_probs3_reduced = log_probs3.mean(dim=1)

            ratios1 = torch.exp(log_probs1_reduced - old_log_probs1)
            ratios2 = torch.exp(log_probs2_reduced - old_log_probs2)
            ratios3 = torch.exp(log_probs3_reduced - old_log_probs3)


        # print("ratios1.shape",ratios1.shape)
        surr1 = ratios1 * advantages.unsqueeze(-1)
        surr2 = ratios2 * advantages.unsqueeze(-1)
        surr3 = ratios3 * advantages.unsqueeze(-1)

        # Clipped objective to maintain stable training
        clipped_surr1 = torch.clamp(ratios1, 1 - self.clip_range, 1 + self.clip_range) * advantages.unsqueeze(-1)
        clipped_surr2 = torch.clamp(ratios2, 1 - self.clip_range, 1 + self.clip_range) * advantages.unsqueeze(-1)
        clipped_surr3 = torch.clamp(ratios3, 1 - self.clip_range, 1 + self.clip_range) * advantages.unsqueeze(-1)

        # PPO loss
        ppo_loss1 = -torch.min(surr1, clipped_surr1).mean().to(dtype=torch.bfloat16)
        ppo_loss2 = -torch.min(surr2, clipped_surr2).mean().to(dtype=torch.bfloat16)
        ppo_loss3 = -torch.min(surr3, clipped_surr3).mean().to(dtype=torch.bfloat16)

        sum_sum=0
        for i in range(0,len(decoded_labels2)):
            # print(decoded_labels2)
            # print("len(decoded_labels2)-1",len(decoded_labels2)-1)
            # Extract entities
            # print("rag[i]",rag[i])
            entities_text1 = extract_entities(rag[i])
            entities_text2 = extract_entities(decoded_preds2[i])
             # Calculate overlap
            overlap = entities_text1.intersection(entities_text2)
            union_comb=entities_text1.union(entities_text2)

            try:
                rl_reward_decoder_2 = len(overlap) / len(union_comb)
            except:
                rl_reward_decoder_2 = 0
            
            entities_text2 = extract_entities(decoded_preds3[i])

            # Calculate overlap
            overlap = entities_text1.intersection(entities_text2)
            
            union_comb=entities_text1.union(entities_text2)

            try:
                rl_reward_decoder_3 = len(overlap) / len(union_comb)
            except:
                rl_reward_decoder_3 = 0
            # print("rl_reward_decoder_3",rl_reward_decoder_3)
            
            values=[]   
    
            values.append(rl_reward_decoder_2 ) #Decrease the loss
            values.append(rl_reward_decoder_3 ) #Decrease the loss
            result = weighted_average(values)
            sum_sum+=result


        sum_sum=sum_sum/len(decoded_labels1)


        # Total loss

        RLloss = (ppo_loss1 + ppo_loss2 + ppo_loss3) / 3.0
        RLloss = normalise_tanh(RLloss)



        # print("CHECK 1!")
        loss_fct = nn.CrossEntropyLoss()
        assert labels1.min() >= 0 and labels1.max() < self.model.config.vocab_size
        assert labels2.min() >= 0 and labels2.max() < self.model.config.vocab_size
        assert labels3.min() >= 0 and labels3.max() < self.model.config.vocab_size

        try:
            
            loss1 = loss_fct(logits1.view(-1,self.model.config.vocab_size),labels1.view(-1))
            # loss1 = loss_fct(preds1.float(),labels1.float())
            # print("loss1: ",loss1)
            loss2 = loss_fct(logits2.view(-1,self.model.config.vocab_size),labels2.view(-1))
            # loss2 = loss_fct(preds2.float(),labels2.float())
            # print("loss2: ",loss2)
            loss3 = loss_fct(logits3.view(-1,self.model.config.vocab_size),labels3.view(-1))
        
        except:

            print("except block Slice or truncate to the first 1360 tokens if you only need that many")
            labels1 = labels1.view(2, -1)

            # 2) Slice or truncate to the first 1360 tokens if you only need that many:
            labels1 = labels1[:, :logits1.size()[1]]  # => shape [2, 1360]

            labels2 = labels2.view(2, -1)
            labels2 = labels1[:, :logits2.size()[1]]
            labels3 = labels3.view(2, -1)
            labels3 = labels3[:, :logits3.size()[1]]



            loss1 = loss_fct(logits1.contiguous().view(-1,self.model.config.vocab_size),labels1.contiguous().view(-1))
            # loss1 = loss_fct(preds1.float(),labels1.float())
            # print("loss1: ",loss1)
            loss2 = loss_fct(logits2.contiguous().view(-1,self.model.config.vocab_size),labels2.contiguous().view(-1))
            # loss2 = loss_fct(preds2.float(),labels2.float())
            # print("loss2: ",loss2)
            loss3 = loss_fct(logits3.contiguous().view(-1,self.model.config.vocab_size),labels3.contiguous().view(-1))
  

        # # Aggregate the losses
        loss_mid1 = 0.2*loss1 + 0.6*loss2
        loss = (0.2*loss3 + loss_mid1) / 3
        
        # loss = (0.3*l1 + 0.2*l2 + 0.5*l3)/3
        # print("CHECK 2!")
        # print("l1: ",l1)
        # print("l2: ",l2)
        # print("l3: ",l3)

        # important
        ner_loss=torch.tensor(sum_sum)
        loss=torch.sub(loss,ner_loss)


        if RLloss < loss:
            loss = loss - RLloss
        else:
            scaling_factor = loss / (RLloss + 1e-6)  # Dynamically scale RLloss
            min_l = 0.1 * loss
            loss = loss - (scaling_factor * RLloss)
            loss = torch.clamp(loss, min = min_l)#maintain atleast 10% of original loss
        

        if RLloss < loss:
            scaling_factor = torch.abs(RLloss  / ( loss + 1e-6))
            # print("if ot scaling_factor:",scaling_factor)
            # loss = loss - RLloss
            min_l = 0.1 * loss
            # print("if ot loss:", loss)
            # print("if scaling_factor * RLloss: ",(scaling_factor) * RLloss)
            loss = loss - (scaling_factor * RLloss)
            if loss <= 0.0:
                loss = torch.clamp(loss, min = min_l)
            # print(f"If'ot Ner Loss: {ner_loss},Rl loss: {RLloss}, Loss1: {loss1}, Loss2: {loss2}, Loss3: {loss3}, Total Loss: {loss}")
            
        else:
            scaling_factor = torch.abs(loss / (RLloss + 1e-6))  # Dynamically scale RLloss
            # print("else ot scaling_factor:",scaling_factor)
            min_l = 0.1 * loss
            # print("else ot loss:", loss)
            loss = loss - (scaling_factor * RLloss)
            # print("else scaling_factor * RLloss: ",scaling_factor * RLloss)
            # print(f"else'ot Ner Loss: {ner_loss},Rl loss: {RLloss}, Loss1: {loss1}, Loss2: {loss2}, Loss3: {loss3}, Total Loss: {loss}")
            
            loss = torch.clamp(loss, min = min_l)#maintain atleast 10% of original loss
        

        return loss
        
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1