
import pickle
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
# import nltk
# nltk.download("punkt")
from transformers import Qwen2VLProcessor
os.environ['WANDB_DISABLED']="True"
os.environ['WANDB_MODE']="offline"
sys.path.insert(0, os.getcwd()+'/custom_transform')
# print("check")
from modeling_phi3_v import Phi3VForCausalLM
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

from custom_loss_si_NOblipNerPPObuf_valuehead_fixLabel_phi import CustomTrainer
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

max_input_length = 1500 #2824 #265210 #1024 #264192 #2600 #4600    #256
num_epochs = 3
path = "Parent Folder Path"
model = "phi3v"
config = model+"NERNOPPO_md_full_modelargmax"+str(max_input_length)
filename_model= config+"_ep_"+str(num_epochs)
print(filename_model)
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
    # print("l: ", l[0])
    l = l[0]
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end
    # print("list(zip(start_indexes, end_indexes)): ",list(zip(start_indexes, end_indexes)))
    return list(zip(start_indexes, end_indexes))

min_pixels = 256*28*28
max_pixels = 1024*28*28
# m_id = "Qwen/Qwen2-VL-2B-Instruct"
m_id = "microsoft/Phi-3.5-vision-instruct" 

# Model and Processor Setup
model = Phi3VForCausalLM.from_pretrained(
# model = Qwen2VLForConditionalGeneration.from_pretrained(
    m_id,
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16,
    _attn_implementation="eager",
).to(device)

model = get_peft_model(model, peft_config)
processor = AutoProcessor.from_pretrained(m_id,min_pixels=min_pixels, max_pixels=max_pixels,trust_remote_code=True,num_crops=16 )
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
rag_output_path = path+"rag_similarity_docs_title_rouge_summ/"
video_embed_path = path+"jpgs/"

test_vid_embed =[]
def fetch_image(image_path):
    try:
        image_obj = Image.open(image_path)
        # print("image_path:",image_path)
        return image_path
    except:
        # print("Black")
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
                 {
                    "role": "system",
                    "content": "You are an expert English question annotator responsible for annotating a dataset."
                },
        {
            "role": "user",
            "content": f"""
                        <|image_1|>\n
                        <|image_2|>\n
                        {vanilla_inputs},
            """,
        },
        
    ]   

            vanilla_label = f'{qlabel}<|end|>\n<|endoftext|>'
            # vanilla_inputs = processor.apply_chat_template(
            vanilla_inputs = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

            # print("vanilla_inputs: ",vanilla_inputs)
            # Collect inputs
            text_inputs.append(vanilla_inputs)

            prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""
   
            decoder_2_inputs = str(" Topic: "+str(topic)+" "+str(image_text))+rag
            decoder_2_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
            
            decoder_2_inputs = prompt_short+truncate_text(decoder_2_inputs)

            messages = [
                 {
                    "role": "system",
                    "content": "You are an expert English question annotator responsible for annotating a dataset."
                },
        {
            "role": "user",
            "content": f"""
                        <|image_1|>\n
                        <|image_2|>\n
                        {decoder_2_inputs},
            """,
        },
    ]            

            gpt_label = f'{gpt_qs}<|end|>\n<|endoftext|>'
            decoder_2_inputs = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
            # Collect inputs
            text_inputs_gpt.append(decoder_2_inputs)

            prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""
   
            decoder_3_inputs = str(" Topic: "+str(topic)+" "+str(image_text))+rag
            decoder_3_inputs = replace_and_merge_xml_tags(str(image_text),text_1,text_2)
            decoder_3_inputs = prompt_short+truncate_text(decoder_3_inputs)

            messages = [
                 {
                    "role": "system",
                    "content": "You are an expert English question annotator responsible for annotating a dataset."
                },
        {
            "role": "user",
            "content": f"""
                        <|image_1|>\n
                        <|image_2|>\n
                        {decoder_3_inputs},
            """,
        },

    ]            

            # prompt_short="""You have two images and text passages. Create one multi-hop question that meaningfully connects or compares the images and/or passages."""
            gemini_label = f'{gemini_qs}<|end|>\n<|endoftext|>'
            decoder_3_inputs = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
            text_inputs_gemini.append(decoder_3_inputs)
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
        labels_RAG = tokenizer(rag_out, max_length=max_input_length, truncation=True)
    
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
                 {
                    "role": "system",
                    "content": "You are an expert English question annotator responsible for annotating a dataset."
                },
        {
            "role": "user",
            "content": f"""
                        <|image_1|>\n
                        <|image_2|>\n
                        {vanilla_inputs},
            """,
        },

    ]            

        
            vanilla_label = f'{qlabel}<|end|>\n<|endoftext|>'
            vanilla_inputs = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
            text_inputs.append(vanilla_inputs)
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
                 {
                    "role": "system",
                    "content": "You are an expert English question annotator responsible for annotating a dataset."
                },
        {
            "role": "user",
            "content": f"""
                        <|image_1|>\n
                        <|image_2|>\n
                        {vanilla_inputs},
            """,
        },

    ]            


            vanilla_label = f'{qlabel}<|end|>\n<|endoftext|>'

            vanilla_inputs = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
            text_inputs.append(vanilla_inputs)
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

# raw_datasets = transform_dialogsumm_to_huggingface_dataset(dataset_train[:20],dataset_val[:2],dataset_test[:2])

from datasets import Features, Sequence, Value
tokenized_datasets = raw_datasets["train"].map(preprocess_function,  batched=True, batch_size=1)
# print("tokenized_datasets train done")
tokenized_datasets_test = raw_datasets["test"].map(preprocess_function_test,  batched=True, batch_size=1)
# print("tokenized_datasets test done")
tokenized_datasets_val1 = raw_datasets["validation"].map(preprocess_function_val, batched=True, batch_size=1)
# print("tokenized_datasets val done")

import time
from trl import SFTConfig
# pip install -U trl #trl==0.13.0
# pip install --upgrade triton #triton==3.1.0


label_names=['labels','gemini','gpt4v',"rag"]
batch_size = 1


# args = Seq2SeqTrainingArguments(
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
        
        labels_vanilla = [item['vanilla_label'] for item in batch]
        labels_gpt4 = [item['gpt_label'] for item in batch]
        labels_gemini = [item['gemini_label'] for item in batch]

        # print("inputs_vanilla:", inputs_vanilla[0])
        inputs_gpt4 = [item['gpt4_inputs_input_ids'] for item in batch]
        inputs_gemini = [item['gemini_inputs_input_ids'] for item in batch]
        for idx, message in enumerate(inputs_vanilla):
            # image_inputs_list.append([Image.open(image_1[idx]).resize((256, 256)).convert("RGB"),Image.open(image_2[idx]).resize((256, 256)).convert("RGB")])
            # image_inputs_list.append([image_1[idx],image_2[idx]])
            image_inputs_list.append(Image.open(image_1[idx]).resize((256, 256)).convert("RGB"))
            image_inputs_list.append(Image.open(image_2[idx]).resize((256, 256)).convert("RGB"))
          
        
        #
        
        # Tokenize all text inputs for the batch
        # print("image_inputs_list:",image_inputs_list)
        # assert len(inputs_vanilla) == len(image_inputs_list), "Mismatch in text and image inputs"

        vanilla_inputs = processor(
            text=inputs_vanilla[0],
            images=image_inputs_list,
            # videos=None,
            padding=True,
            return_tensors="pt",
            truncation=True,       # Enable truncation
            max_length=max_input_length, 
        )

        
        ######################################

        gpt4_inputs = processor(
        text=inputs_gpt4[0],
        images=image_inputs_list,
        # videos=None,
        padding=True,
        return_tensors="pt",
        truncation=True,       # Enable truncation
        max_length=max_input_length, 
    )
        # inputs_gpt4 = gpt4_inputs["input_ids"].tolist()
        # mask_gpt4 = gpt4_inputs["attention_mask"].tolist()

        # Do not add bos token to answer
        labels_gpt4_input_ids =  processor.tokenizer(
            labels_gpt4, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids_gpt4 = torch.cat([gpt4_inputs["input_ids"], labels_gpt4_input_ids], dim=1)
        ignore_index = -100
        labels_gpt = torch.cat(
            [
                torch.tensor([ignore_index] * len(gpt4_inputs["input_ids"][0])).unsqueeze(0),
                labels_gpt4_input_ids,
            ],
            dim=1,
        )


        gpt4_inputs["input_ids"] = input_ids_gpt4
        gpt4_inputs["labels"] = labels_gpt
        del gpt4_inputs['attention_mask']

        ######################################

        gemini_inputs = processor(
        text=inputs_gemini[0],
        images=image_inputs_list,
        # videos=None,
        padding=True,
        return_tensors="pt",
        truncation=True,       # Enable truncation
        max_length=max_input_length, 
    )
        # inputs_gemini = gemini_inputs["input_ids"].tolist()
        # mask_gemini = gemini_inputs["attention_mask"].tolist()


        # Do not add bos token to answer
        labels_gemini_input_ids =  processor.tokenizer(
            labels_gemini, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids_gemini = torch.cat([gemini_inputs["input_ids"], labels_gemini_input_ids], dim=1)
        ignore_index = -100
        labels_gem = torch.cat(
            [
                torch.tensor([ignore_index] * len(gemini_inputs["input_ids"][0])).unsqueeze(0),
                labels_gemini_input_ids,
            ],
            dim=1,
        )


        gemini_inputs["input_ids"] = input_ids_gemini
        gemini_inputs["labels"] = labels_gem
        del gemini_inputs['attention_mask']

        ######################################
        
        flag = 0
    except Exception as e:
        inputs_vanilla = [item['input_ids'] for item in batch]
        for idx, message in enumerate(inputs_vanilla):
            # image_inputs_list.append([Image.open(image_1[idx]).resize((256, 256)).convert("RGB"),Image.open(image_2[idx]).resize((256, 256)).convert("RGB")])
            
            image_inputs_list.append(Image.open(image_1[idx]).resize((256, 256)).convert("RGB"))
            image_inputs_list.append(Image.open(image_2[idx]).resize((256, 256)).convert("RGB"))
          
        
            
            # image_inputs_list.append([image_1[idx],image_2[idx]])
        # Tokenize all text inputs for the batch
        # assert len(inputs_vanilla) == len(image_inputs_list), "Mismatch in text and image inputs"

        vanilla_inputs = processor(
            text=inputs_vanilla[0],
            images=image_inputs_list,
            # videos=None,
            padding=True,
            return_tensors="pt",
            truncation=True,       # Enable truncation
            max_length=max_input_length, 
        )
        inputs_vanilla = vanilla_inputs["input_ids"].tolist()
        inputs_gpt4 = inputs_vanilla
        inputs_gemini = inputs_vanilla
        # mask_vanilla = vanilla_inputs["attention_mask"].tolist()
        # mask_gpt4 = mask_vanilla
        # mask_gemini = mask_vanilla
        flag = 1

    # Do not add bos token to answer
    labels_vanilla_input_ids =  processor.tokenizer(
        labels_vanilla, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
    input_ids_vanilla = torch.cat([vanilla_inputs["input_ids"], labels_vanilla_input_ids], dim=1)
    ignore_index = -100
    labels_van = torch.cat(
            [
                torch.tensor([ignore_index] * len(vanilla_inputs["input_ids"][0])).unsqueeze(0),
                labels_vanilla_input_ids,
            ],
            dim=1,
        )


    vanilla_inputs["input_ids"] = input_ids_vanilla
    vanilla_inputs["labels"] = labels_van
    del vanilla_inputs['attention_mask']
    
    try:
        labels_rag = [item['rag'] for item in batch]
        flag = 0
    except Exception as e:
        flag = 1
        labels_gemini = labels_van
        labels_rag = labels_van
        labels_gpt = labels_van

    labels_text_rag = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(lst).clone().detach() for lst in labels_rag], batch_first=True, padding_value=0)
    
    if flag == 1:
        flag = 0
        # return {'input_ids': inputs_text_vanilla, 'labels': labels_text,'attention_mask': attention_mask_vanilla}
        return vanilla_inputs
    else:

    
        return {'vanilla_inputs_input_ids': vanilla_inputs["input_ids"], 
        'gpt4_inputs_input_ids': gpt4_inputs["input_ids"],
        'gemini_inputs_input_ids': gemini_inputs["input_ids"], 
        'rag': labels_text_rag, 
        'gpt4v': gpt4_inputs["labels"], 
        'gemini': gemini_inputs["labels"], 
        'labels': vanilla_inputs["labels"],  
        "pixel_values": vanilla_inputs["pixel_values"],
        "image_sizes": vanilla_inputs["image_sizes"],

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
                 {
                    "role": "system",
                    "content": "You are an expert English question annotator responsible for annotating a dataset."
                },
        {
            "role": "user",
            "content": f"""
                        <|image_1|>\n
                        <|image_2|>\n
                        {vanilla_inputs},
            """,
        },

    ]            


                vanilla_inputs = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
                # image_inputs, video_inputs = process_vision_info(messages)
                text_inputs.append(vanilla_inputs)
                image_inputs_list.append(Image.open(i_1).resize((256, 256)).convert("RGB"))
                image_inputs_list.append(Image.open(i_2).resize((256, 256)).convert("RGB"))


                inputs = processor(
                    text=text_inputs[0],
                    images=image_inputs_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_length,
                ).to("cuda")

                inputs = inputs.to(model.device)

                generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64
        )

                decoded_preds = processor.batch_decode(
                    generated_ids[:, inputs['input_ids'].size(1) :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            

                print("Generated Question:", decoded_preds)
                final_decoded_preds.append(decoded_preds)

    return final_decoded_preds

# Generate and save predictions
predictions = generate_predictions(dataset_test)

# open file
with open(MODEL_PATH+'output.txt', 'w+') as f:
    
    # write elements of list
    for items in final_decoded_preds:
        f.write('%s\n' %items)
    
    print("File written successfully")


# close the file
f.close()    
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
    