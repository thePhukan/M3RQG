
import pickle
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import json
from datasets import load_metric,Dataset,DatasetDict
from datasets import Features, Sequence, Value
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer,TrainingArguments
from transformers import AutoTokenizer,Phi3ForCausalLM,AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import pandas as pd
from tqdm import tqdm
from transformers import Trainer
from copy import deepcopy
from transformers import AutoProcessor
torch.cuda.empty_cache()
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
# import wandb
import torch.nn.functional as F
import sys

os.environ['WANDB_DISABLED']="True"
os.environ['WANDB_MODE']="offline"
sys.path.insert(0, os.getcwd()+'/custom_transform')

from modeling_llava import LlavaForConditionalGeneration 
from trl import SFTTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print("\nDEVICE:\t",device)
metric = load_metric("rouge.py",trust_remote_code=True) 

from custom_loss_si_NOblipNerPPObuf_valuehead_fixLabel_llava import CustomTrainer
TEST_SUMMARY_ID = 1
##############################################################################
def transform_single_dialogsumm_file(file):
    result = {"Guid":[],"topic":[],"image_text":[],"Q":[], "First_Image": [],"Second_Image": [], "A": [],"Qcate": [],"gemini": [],"gpt4v":[]}

    for i in range(len(file)):
        # print(file["Guid"][i])
        result["Guid"].append(file["Guid"][i])
        result["topic"].append(file["topic"][i])
        result["Q"].append(file["Q"][i])
        result["First_Image"].append(str(file["First_Image"][i]))
        result["Second_Image"].append(str(file["Second_Image"][i]))
        result["image_text"].append(str(file["image_text"][i]))
        result["A"].append(str(file["A"][i]))
        result["Qcate"].append(str(file["Qcate"][i]))
        result["gemini"].append(str(file["gemini"][i]))
        result["gpt4v"].append(str(file["gpt4v"][i]))
    return Dataset.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(train,validation,test):
    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_single_dialogsumm_file(test)
    return DatasetDict({"train":train,"validation":validation, "test": test})

max_input_length = 2000 
num_epochs = 3

path = "Parent Folder Path"
# path = os.getcwd()+"/"
# print("File location:", path)
# model = "qwen2vl_2B_inst"
# model = "phi3v"
model = "llava"
config = model+"NONERNOPPO_md_full_r1024bf16_pmtstrct_fixedback_modelargmax"+str(max_input_length)
filename_model= config+"_ep_"+str(num_epochs)
print(filename_model)
MODEL_PATH_CHECKPOINT = path+"Model Path/"+filename_model+"_Loss_Checkpoints.pt"

MODEL_PATH = path+"/Model Path/"+filename_model


is_cuda = torch.cuda.is_available()
############ WEBQA ############
import glob
import random, time, base64
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
dataset_val = pd.read_excel(path+'dataset_split/dataset_val_gem_gpt4.xlsx')
print("dataset_val:",len(dataset_val))
dataset_train = pd.read_excel(path+'dataset_split/dataset_train_gem_gpt4.xlsx')
dataset_test = pd.read_excel(path+'dataset_split/dataset_test_gem_gpt4.xlsx')
print("dataset_train:",len(dataset_train))
print("dataset_test:",len(dataset_test))


def find_assistant_content_sublist_indexes(l):
    """
    l is assumed to be a list with one element containing the tokenized message.
    
    For llava-hf/llava-1.5-7b-hf using the following prompt template:
    
    vanilla_inputs:  USER: <image>
    <image>
    ... (other text, image captions, etc.) ...
    Rag:   ASSISTANT: "What color do the calla lily and the columbine flower have in common?"
    
    This function finds the index boundaries in the token list for the assistant’s reply.
    It looks for the token sequence corresponding to "ASSISTANT:" and then uses the next occurrence
    of "USER:" (if any) as the end marker. If no subsequent "USER:" is found, the assistant reply
    is assumed to run until the end of the token list.
    """
    # Unpack the single-element list
    tokens = l[0]
    
    # Dynamically get token ids for the markers.
    assistant_marker = "ASSISTANT: "
    assistant_marker_ids = processor.tokenizer.encode(assistant_marker)
    # print("Assistant marker tokens:", assistant_marker_ids)
    # We'll optionally use "USER:" as an end marker if multiple turns exist.
    user_marker = "USER:"
    user_marker_ids = processor.tokenizer.encode(user_marker)
    
    start_indexes = []
    end_indexes = []
    marker_len = len(assistant_marker_ids)
    # print("CHECK")
    # Search for every occurrence of the assistant marker in the token list.
    for i in range(len(tokens) - marker_len + 1):
        if tokens[i:i+marker_len] == assistant_marker_ids:
            # The assistant content starts immediately after the marker.
            print("\n\nFound assistant response start\n\n")
            start_idx = i + marker_len
            
            # Look for a subsequent USER: marker as an end boundary.
            end_idx = None
            for j in range(start_idx, len(tokens) - len(user_marker_ids) + 1):
                if tokens[j:j+len(user_marker_ids)] == user_marker_ids:
                    end_idx = j
                    break
            # If no USER: marker is found, assume the reply goes until the end.
            if end_idx is None:
                end_idx = len(tokens)
                
            start_indexes.append(start_idx)
            end_indexes.append(end_idx)
    
    return list(zip(start_indexes, end_indexes))

def find_assistant_indexes_by_decoding(vanilla_inputs):
    """
    Searches the decoded text for the marker "ASSISTANT:".
    Returns a list with a tuple of (start_token_index, end_token_index) for the assistant reply.
    The reply is assumed to start right after "ASSISTANT:" and ends at the next "USER:" marker (if any),
    or at the end of the text.
    """
    # Decode the full text from input_ids (assuming batch size 1)
    full_text = processor.tokenizer.decode(vanilla_inputs["input_ids"][0])
    
    # Find the starting position of the assistant marker in the decoded text.
    marker = "ASSISTANT:"
    start_char = full_text.find(marker)
    if start_char == -1:
        # If not found, return empty list.
        return []
    
    # The assistant reply starts immediately after the marker.
    assistant_reply_start_char = start_char
    
    # Look for the next "USER:" marker after the assistant marker.
    user_marker = "USER:"
    end_char = full_text.find(user_marker, assistant_reply_start_char)
    if end_char == -1:
        end_char = len(full_text)
    
    # Convert the character positions back to approximate token indexes.
    # This is done by re-encoding the text up to these character positions.
    pre_reply_text = full_text[:assistant_reply_start_char]
    assistant_reply_text = full_text[assistant_reply_start_char:end_char]
    
    pre_reply_tokens = processor.tokenizer.encode(pre_reply_text)
    reply_tokens = processor.tokenizer.encode(assistant_reply_text)
    
    start_token_idx = len(pre_reply_tokens)
    end_token_idx = start_token_idx + len(reply_tokens)
    
    return [(start_token_idx, end_token_idx)]


min_pixels = 256*28*28
max_pixels = 1024*28*28


###### CHANGE IN THE BACKEND AS WELL!!!!!!! ########
# m_id = "microsoft/Phi-3.5-vision-instruct" 
m_id = "llava-hf/llava-1.5-7b-hf"
###### CHANGE IN THE BACKEND AS WELL!!!!!!!! ########


# Model and Processor Setup
model = LlavaForConditionalGeneration.from_pretrained(
    m_id,
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True,
).to(device)

model = get_peft_model(model, peft_config)
processor = AutoProcessor.from_pretrained(m_id)
tokenizer = processor.tokenizer
tokenizer.padding_side = "right"


print(model.print_trainable_parameters())

def replace_non_english_with_space(text):
    # Use regex to match any character that is not a-z, A-Z, 0-9, or common punctuation
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,!?\'\"\s]', ' ', text)
    # Remove any excessive spaces that may result from replacements
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def truncate_text(text, max_words=1000):
    # Split the text into words
    # cleaned_text = replace_non_english_with_space(text)
    cleaned_text = text
    # print(cleaned_text)
    words = cleaned_text.split()
    # print(words)
    word_count = len(words)
    # print("word_count",word_count)
    # If word count exceeds the limit, truncate the text
    if word_count > max_words:
        truncated_text = ' '.join(words[:max_words])
        # print(f"Text truncated to {max_words} words.")
    else:
        truncated_text = str(cleaned_text)
    return truncated_text


def replace_and_merge_xml_tags(text, text_1, text_2):
    replacements = {
        r"<First_Image>": f"First Image Text: ",
        r"<\\First_Image>": f". Rag: {text_1} ",
        r"<Second_Image>": f"Second Image Text: ",
        r"<\\Second_Image>": f". Rag: {text_2} "
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    # print("replace_and_merge_xml_tags:", text)
    return text

max_target_length = 1024
import ast
rag_output_path = path+"rag_docs/"
video_embed_path = path+"jpgs/"

test_vid_embed =[]
def fetch_image(image_path):
    try:
        image_obj = Image.open(image_path)
        return image_path
    except:
        black_image = path+"jpgs/black_image.jpg"
        return black_image

def preprocess_function(examples):
    inputs =[]
    inputs_decoder_2 =[]
    inputs_decoder_3 =[]
    compressed_vid = []
    image1 =[]
    image2=[]
    model_inputs={}
    input_embedd_feat = []
    rag_out = []
    prompt_short="""Generate an english question that meaningfully connects or compares the images and/or passages."""
    
    vanila_q_template = []
    gemini_q_template = []
    gpt4_q_template = []
    jpgs = []
    # Initialize lists to hold tokenized inputs
    text_inputs = []
    image_inputs_list = []
    video_inputs_list = []

    text_inputs_gpt = []
    image_inputs_list_gpt = []
    video_inputs_list_gpt = []

    text_inputs_gemini = []
    image_inputs_list_gemini = []
    video_inputs_list_gemini = []

    for (topic, Qcate, image_text, First_Image, Second_Image, qlabel, gemini_qs, gpt_qs) in zip(examples["topic"], 
    examples["Qcate"], examples["image_text"], examples["First_Image"], 
    examples["Second_Image"], examples["Q"], examples["gemini"], examples["gpt4v"]):
            i_1 = fetch_image(video_embed_path + str(First_Image) + ".jpg")
            i_2 = fetch_image(video_embed_path + str(Second_Image) + ".jpg")
            text_1 = ""
            text_2 = ""
            try:
                docs = sorted(glob.glob(rag_output_path+str(First_Image)+"/*.txt"), reverse=True)
                for text_path in docs:
                    with open(text_path,"r",encoding = "utf-8") as f1:
                        text = f1.read()

                    text_1 = text_1 + str(text)
            except Exception as e:
                text_1 = "NA"
            try:
                docs = sorted(glob.glob(rag_output_path+str(Second_Image)+"/*.txt"), reverse=True)
                for text_path in docs:
                    with open(text_path,"r",encoding = "utf-8") as f2:
                        text = f2.read()

                    text_2 = text_2 + str(text)
                
            except Exception as e:
                text_2 = "NA"
                
            rag=str("First Image Text:"+text_1+" "+" Second Image Text:"+text_2+" ")
            vanilla_inputs = str(" Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+rag
            vanilla_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
            
            vanilla_inputs = prompt_short+truncate_text(vanilla_inputs)


            messages = [
                 f"USER: <image>\n<image>\nYou are an expert English question annotator responsible for annotating a dataset. {vanilla_inputs}\nASSISTANT: {qlabel}"

            ]

            vanilla_label = f'{qlabel}<|end|>\n<|endoftext|>'
       
            text_inputs.append(messages)


            prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""
   
            decoder_2_inputs = str(" Topic: "+str(topic)+" "+str(image_text))+rag
            decoder_2_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
            
            decoder_2_inputs = prompt_short+truncate_text(decoder_2_inputs)
        

            messages = [
                 f"USER: <image>\n<image>\nYou are an expert English question annotator responsible for annotating a dataset. {decoder_2_inputs}\nASSISTANT: {gpt_qs}"

            ]
            gpt_label = f'{gpt_qs}<|end|>\n<|endoftext|>'
            text_inputs_gpt.append(messages)
            prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""
   
            decoder_3_inputs = str(" Topic: "+str(topic)+" "+str(image_text))+rag
            decoder_3_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
            decoder_3_inputs = prompt_short+truncate_text(decoder_3_inputs)
            
            messages = [
                 f"USER: <image>\n<image>\nYou are an expert English question annotator responsible for annotating a dataset. {decoder_3_inputs}\nASSISTANT: {gemini_qs}"

            ]
            

            gemini_label = f'{gemini_qs}<|end|>\n<|endoftext|>'

            text_inputs_gemini.append(messages)
            image1.append(str(fetch_image(video_embed_path + str(First_Image) + ".jpg")))
            image2.append(str(fetch_image(video_embed_path + str(Second_Image) + ".jpg")))
            rag_out.append(rag)
            compressed_vid.append(torch.zeros(1,1))
    image_inputs = image_inputs_list
    video_inputs = None
    video_inputs_gpt = None
    image_inputs_gpt = image_inputs_list_gpt
    video_inputs_gemini = None
    image_inputs_gemini = image_inputs_list_gemini
    model_inputs = {
        "vanilla_inputs_input_ids": text_inputs,
        # "vanilla_inputs_attention_mask": None,
        "gpt4_inputs_input_ids": text_inputs_gpt,
        # "gpt4_inputs_attention_mask": None,
        "gemini_inputs_input_ids": text_inputs_gemini,
    }
    label_input =[]
    gQs_label_input = []

    with tokenizer.as_target_tokenizer():
        labels_RAG = tokenizer(rag_out, max_length=1024, truncation=True)
    
    model_inputs["rag"] = labels_RAG["input_ids"]
    model_inputs["image_id_1"] = image1
    model_inputs["image_id_2"] = image2
    model_inputs['vanilla_label'] = [vanilla_label]
    model_inputs['gpt_label'] = [gpt_label]
    model_inputs['gemini_label'] = [gemini_label]
    return model_inputs

def preprocess_function_val(examples):
    inputs = []
    compressed_vid = []
    model_inputs = {}
    input_embedd_feat = []
    rag_out = []

    prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""

    jpgs = []
    # Initialize lists to hold tokenized inputs
    text_inputs = []
    image_inputs_list = []
    video_inputs_list = []
    image1 = []
    image2 = []

    for (topic, Qcate, image_text, First_Image, Second_Image,qlabel) in zip(examples["topic"], examples["Qcate"], examples["image_text"], examples["First_Image"], examples["Second_Image"], examples["Q"]):
            i_1 = fetch_image(video_embed_path + str(First_Image) + ".jpg")
            i_2 = fetch_image(video_embed_path + str(Second_Image) + ".jpg")
            text_1 = ""
            text_2 = ""
            try:
                docs = sorted(glob.glob(rag_output_path+str(First_Image)+"/*.txt"), reverse=True)
                for text_path in docs:
                    with open(text_path,"r",encoding = "utf-8") as f1:
                        text = f1.read()

                    text_1 = text_1 + str(text)
            except Exception as e:
                text_1 = "NA"
            try:
                docs = sorted(glob.glob(rag_output_path+str(Second_Image)+"/*.txt"), reverse=True)
                for text_path in docs:
                    with open(text_path,"r",encoding = "utf-8") as f2:
                        text = f2.read()

                    text_2 = text_2 + str(text)
                
            except Exception as e:
                text_2 = "NA"
                
            rag=str("First Image Text:"+text_1+" "+" Second Image Text:"+text_2+" ")
            vanilla_inputs = str(" Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+rag
            vanilla_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
            vanilla_inputs = prompt_short+truncate_text(vanilla_inputs)
            

            messages = [
                 f"USER: <image>\n<image>\nYou are an expert English question annotator responsible for annotating a dataset. {vanilla_inputs}\nASSISTANT: {qlabel}"

            ]

            vanilla_label = f'{qlabel}<|end|>\n<|endoftext|>'

            text_inputs.append(messages)



            image1.append(str(fetch_image(video_embed_path + str(First_Image) + ".jpg")))
            image2.append(str(fetch_image(video_embed_path + str(Second_Image) + ".jpg")))
            rag_out.append(rag)
            # compressed_vid.append(torch.zeros(1,1))
    # video_inputs = None
    model_inputs = {
        "input_ids": text_inputs,
    }

    
    model_inputs["image_id_1"] = image1
    model_inputs["image_id_2"] = image2
    model_inputs['vanilla_label'] = [vanilla_label]
    return model_inputs

def preprocess_function_test(examples):
    inputs = []
    compressed_vid = []
    model_inputs = {}
    input_embedd_feat = []
    rag_out = []

    prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""

    jpgs = []
    text_inputs = []
    image_inputs_list = []
    video_inputs_list = []
    image1=[]
    image2=[]
    for (topic, Qcate, image_text, First_Image, Second_Image, qlabel) in zip(examples["topic"], 
    examples["Qcate"], examples["image_text"], examples["First_Image"], 
    examples["Second_Image"], examples["Q"]):
    
            i_1 = fetch_image(video_embed_path + str(First_Image) + ".jpg")
            i_2 = fetch_image(video_embed_path + str(Second_Image) + ".jpg")
            text_1 = ""
            text_2 = ""
            try:
                docs = sorted(glob.glob(rag_output_path+str(First_Image)+"/*.txt"), reverse=True)
                for text_path in docs:
                    with open(text_path,"r",encoding = "utf-8") as f1:
                        text = f1.read()
                    text_1 = text_1 + str(text)
            except Exception as e:
                text_1 = "NA"

            try:
                docs = sorted(glob.glob(rag_output_path+str(Second_Image)+"/*.txt"), reverse=True)
                for text_path in docs:
                    with open(text_path,"r",encoding = "utf-8") as f2:
                        text = f2.read()

                    text_2 = text_2 + str(text)
                
            except Exception as e:
                text_2 = "NA"

            rag=str("First Image Text:"+text_1+" "+" Second Image Text:"+text_2+" ")
            vanilla_inputs = str(" Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+rag
            vanilla_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
            vanilla_inputs = prompt_short+truncate_text(vanilla_inputs)

            messages = [
                 f"USER: <image>\n<image>\nYou are an expert English question annotator responsible for annotating a dataset. {vanilla_inputs}\nASSISTANT: {qlabel}"

            ]

            vanilla_label = f'{qlabel}<|end|>\n<|endoftext|>'

            text_inputs.append(messages)


            image1.append(str(fetch_image(video_embed_path + str(First_Image) + ".jpg")))
            image2.append(str(fetch_image(video_embed_path + str(Second_Image) + ".jpg")))

            rag_out.append(rag)
            # compressed_vid.append(torch.zeros(1,1))
    # video_inputs = None
    model_inputs = {
        "input_ids": text_inputs,
    }

    label_input = []
    model_inputs["image_id_1"] = image1
    model_inputs["image_id_2"] = image2
    model_inputs['vanilla_label'] = [vanilla_label]
    return model_inputs

raw_datasets = transform_dialogsumm_to_huggingface_dataset(dataset_train,dataset_val,dataset_test)

from datasets import Features, Sequence, Value
tokenized_datasets = raw_datasets["train"].map(preprocess_function,  batched=True, batch_size=1)

tokenized_datasets_test = raw_datasets["test"].map(preprocess_function_test,  batched=True, batch_size=1)

tokenized_datasets_val1 = raw_datasets["validation"].map(preprocess_function_val, batched=True, batch_size=1)


import time
from trl import SFTConfig
# pip install -U trl #trl==0.13.0
# pip install --upgrade triton #triton==3.1.0


label_names=['labels','gemini','gpt4v',"rag"]
batch_size = 1

args = SFTConfig(
# args = TrainingArguments(
    config+"rog_ep_"+str(num_epochs),
    learning_rate=2e-4,  # Learning rate for training
    optim="adamw_torch_fused",  # Optimizer type
    per_device_train_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    gradient_checkpointing=True,
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    save_strategy="epoch",
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    seed=42,
    max_seq_length=max_input_length,
    remove_unused_columns=False,
    label_names = label_names,
    gradient_accumulation_steps=8,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    eval_accumulation_steps=2,
    bf16 =True,
)


def collate_fn(batch):
    flag = 0
    image_inputs_list = []
    image_inputs_list_gemini = []
    image_inputs_list_gpt = []
    image_1 = [item['image_id_1'] for item in batch]
    image_2 = [item['image_id_2'] for item in batch]
    # print("image_2: ",image_2)
    try:
        inputs_vanilla = [item['vanilla_inputs_input_ids'] for item in batch]
        inputs_gpt4 = [item['gpt4_inputs_input_ids'] for item in batch]
        inputs_gemini = [item['gemini_inputs_input_ids'] for item in batch]
        for idx, message in enumerate(inputs_vanilla):
            image_inputs_list.append(Image.open(image_1[idx]).resize((256, 256)).convert("RGB"))
            image_inputs_list.append(Image.open(image_2[idx]).resize((256, 256)).convert("RGB"))
          
        vanilla_inputs = processor(
            text=inputs_vanilla[0],
            images=image_inputs_list,
            # videos=None,
            # padding=True,
            return_tensors="pt",
            truncation=True,       # Enable truncation
            max_length=max_input_length, 
        )

        
        labels = vanilla_inputs["input_ids"].clone()
        labels.fill_(-100)
        assistant_indexes = find_assistant_indexes_by_decoding(vanilla_inputs)
        for begin_idx, end_idx in assistant_indexes:
            
            labels[0, begin_idx:end_idx] = vanilla_inputs["input_ids"][0, begin_idx:end_idx]
        # Assign the labels back to the inputs dictionary
        vanilla_inputs["labels"] = labels


        ######################################

        gpt4_inputs = processor(
        text=inputs_gpt4[0],
        images=image_inputs_list,
        # videos=None,
        # padding=True,
        return_tensors="pt",
        truncation=True,       # Enable truncation
        max_length=max_input_length, 
    )


        labels_gpt = gpt4_inputs["input_ids"].clone()
        labels_gpt.fill_(-100)
        
        assistant_indexes = find_assistant_indexes_by_decoding(gpt4_inputs)
     
        for begin_idx, end_idx in assistant_indexes:

            
            labels_gpt[0, begin_idx:end_idx] = gpt4_inputs["input_ids"][0, begin_idx:end_idx]

        

       
        gpt4_inputs["labels"] = labels_gpt

        ######################################

        gemini_inputs = processor(
        text=inputs_gemini[0],
        images=image_inputs_list,
        return_tensors="pt",
        truncation=True,       # Enable truncation
        max_length=max_input_length, 
    )

        labels_gem = gemini_inputs["input_ids"].clone()
        labels_gem.fill_(-100)
        assistant_indexes = find_assistant_indexes_by_decoding(gemini_inputs)
        
        for begin_idx, end_idx in assistant_indexes:
            
            labels_gem[0, begin_idx:end_idx] = gemini_inputs["input_ids"][0, begin_idx:end_idx]
       
        gemini_inputs["labels"] = labels_gem

        flag = 0
    except Exception as e:
        inputs_vanilla = [item['input_ids'] for item in batch]
        for idx, message in enumerate(inputs_vanilla):
            # image_inputs_list.append([Image.open(image_1[idx]).resize((256, 256)).convert("RGB"),Image.open(image_2[idx]).resize((256, 256)).convert("RGB")])
            
            image_inputs_list.append(Image.open(image_1[idx]).resize((256, 256)).convert("RGB"))
            image_inputs_list.append(Image.open(image_2[idx]).resize((256, 256)).convert("RGB"))
          
        
        
        vanilla_inputs = processor(
            text=inputs_vanilla[0],
            images=image_inputs_list,
            return_tensors="pt",
            truncation=True,       # Enable truncation
            max_length=max_input_length, 
        )
        inputs_vanilla = vanilla_inputs
        inputs_gpt4 = inputs_vanilla
        inputs_gemini = inputs_vanilla
        flag = 1

        labels = vanilla_inputs["input_ids"].clone()
        labels.fill_(-100)
        assistant_indexes = find_assistant_indexes_by_decoding(vanilla_inputs)
    
        for begin_idx, end_idx in assistant_indexes:
            
            labels[0, begin_idx:end_idx] = vanilla_inputs["input_ids"][0, begin_idx:end_idx]
     
        # Assign the labels back to the inputs dictionary
        vanilla_inputs["labels"] = labels


    try:
        labels_rag = [item['rag'] for item in batch]
        flag = 0
    except Exception as e:
        flag = 1
        labels_gemini = labels
        labels_rag = labels
        labels_gpt = labels

    labels_text_rag = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(lst).clone().detach() for lst in labels_rag], batch_first=True, padding_value=-100)
    
    if flag == 1:
        flag = 0
        return vanilla_inputs
    else:

    
        return {'vanilla_inputs_input_ids': vanilla_inputs["input_ids"], 
        'gpt4_inputs_input_ids': gpt4_inputs["input_ids"],
        'gemini_inputs_input_ids': gemini_inputs["input_ids"], 
        'rag': labels_text_rag, 
        'gpt4v': gpt4_inputs["labels"], 
        'gemini': gemini_inputs["labels"], 
        'labels': vanilla_inputs["labels"],  
        'vanilla_inputs_attention_mask': vanilla_inputs["attention_mask"],
        'gpt4_inputs_attention_mask': gpt4_inputs["attention_mask"], 
        'gemini_inputs_attention_mask': gemini_inputs["attention_mask"],
        "pixel_values": vanilla_inputs["pixel_values"],
        # "image_sizes": vanilla_inputs["image_sizes"],
        # "image_grid_thw": image_grid_thw

          }

from transformers import DataCollator,DefaultDataCollator
from rouge_score import rouge_scorer
data_collator = DefaultDataCollator()

import nltk

# Formatting function for the prompts
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}
    end_mapper = {"system": "", "human": "", "gpt": ""}
    for convo in convos:
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in convo)
        texts.append(f"{text}{tokenizer.eos_token}")
    return {"text": texts}


# trainer = SFTTrainer(
trainer = CustomTrainer(
    model,
    args,
    train_dataset= tokenized_datasets,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=tokenizer,

)

# Apply gradient clipping inside training loop
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

trainer.args.max_grad_norm = 10.0  # Equivalent to gradient clipping
trainer.train()
final_decoded_preds=[]


def clean_decoded_text(text):
    text = text.replace("system\n", "").replace("user\n", "").replace("assistant\n", "").strip()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove gibberish
    return text

def generate_predictions(test_dataset):
    model.eval()
    for idx, examples in test_dataset.iterrows():
                # print("examples: ",examples)
                inputs = []
                compressed_vid = []
                model_inputs = {}
                input_embedd_feat = []
                rag_out = []
                prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""
                # prompt_short="""Generate an english question that meaningfully connects or compares the images and/or passages."""
                
                topic = examples["topic"]
                Qcate = examples["Qcate"]
                image_text = examples["image_text"]
                First_Image = examples["First_Image"]
                Second_Image = examples["Second_Image"]
                qlabel = examples["Q"]
                jpgs = []
                text_inputs = []
                image_inputs_list = []
                video_inputs_list = []
                image1=[]
                image2=[]
                i_1 = fetch_image(video_embed_path + str(First_Image) + ".jpg")
                i_2 = fetch_image(video_embed_path + str(Second_Image) + ".jpg")
                text_1 = ""
                text_2 = ""
                try:
                    docs = sorted(glob.glob(rag_output_path+str(First_Image)+"/*.txt"), reverse=True)
                    for text_path in docs:
                        with open(text_path,"r",encoding = "utf-8") as f1:
                            text = f1.read()
                        text_1 = text_1 + str(text)
                except Exception as e:
                    text_1 = "NA"

                try:
                    docs = sorted(glob.glob(rag_output_path+str(Second_Image)+"/*.txt"), reverse=True)
                    for text_path in docs:
                        with open(text_path,"r",encoding = "utf-8") as f2:
                            text = f2.read()

                        text_2 = text_2 + str(text)
                    
                except Exception as e:
                    text_2 = "NA"

                rag=str("First Image Text:"+text_1+" "+" Second Image Text:"+text_2+" ")
                vanilla_inputs = str(" Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+rag
                vanilla_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
                vanilla_inputs = prompt_short+truncate_text(vanilla_inputs)
                

                messages = [
                 f"USER: <image>\n<image>\nYou are an expert English question annotator responsible for annotating a dataset. {vanilla_inputs}\nASSISTANT: "

            ]


                text_inputs.append(messages)
                image_inputs_list.append(Image.open(i_1).resize((256, 256)).convert("RGB"))
                image_inputs_list.append(Image.open(i_2).resize((256, 256)).convert("RGB"))


                inputs = processor(
                    # text=text_inputs[0],
                    text = messages,
                    images=image_inputs_list,
                    return_tensors="pt",
                    # padding=True,
                    truncation=True,
                    max_length=max_input_length,
                )

                inputs = inputs.to(model.device)

                generated_ids = model.generate(**inputs, max_new_tokens=100,do_sample=False,repetition_penalty=1.2)
                
                decoded_preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for text in decoded_preds:
                    decoded_preds = text.split("ASSISTANT:")[-1]
                    
                print("Generated Question:", decoded_preds)
                final_decoded_preds.append(decoded_preds)

    return final_decoded_preds

# Generate and save predictions
predictions = generate_predictions(dataset_test)



# close the file
    
with open(MODEL_PATH+".json", "a") as outfile:    

    outfile.write('[') 
    for index, item in enumerate(tokenized_datasets_test):
        
        dictionary = {"Guid": str(item["Guid"]),"Gold_Question":str(item["Q"]),"Generated_Question":final_decoded_preds[index]}
        # print(dictionary)
        
        if index > 0:
            outfile.write(',')
        json.dump(dictionary, outfile) 
    outfile.write(']') 
    
# print("Trined Successfully! Weights Saved.")
    
print(f"{config+'ep_'+str(num_epochs)}Trined Successfully! Output Saved.")    
    