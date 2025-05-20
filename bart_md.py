import pickle
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # For CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
from datasets import load_metric,Dataset,DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
import os
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm

from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration
from copy import deepcopy
import torch
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import json
# import wandb
import torch.nn.functional as F
# from transformers import pad_sequence
import sys

os.environ['WANDB_DISABLED']="True"
os.environ['WANDB_MODE']="offline"

sys.path.insert(0, os.getcwd()+'/custom_transform')

from modeling_bart_webqa import BartForConditionalGeneration
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,  r=4,  # Reduced rank to lower memory usage
    lora_alpha=16,  # Lowered alpha for reduced scaling of the low-rank adaptation
    lora_dropout=0.2,  # Increased dropout rate to 0.2

)

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print("\nDEVICE:\t",device)
model_checkpoint = "facebook/bart-large"
metric = load_metric("rouge.py",trust_remote_code=True)
from custom_loss_si_NOblipNerPPObuf_valuehead_fixLabel_bart import CustomTrainer


TEST_SUMMARY_ID = 1

##############################################################################
def transform_single_dialogsumm_file(file):
    result = {"Guid":[],"topic":[],"image_text":[],"Q":[], "First_Image": [],"Second_Image": [], "A": [],"Qcate": [],"gemini": [],"gpt4v":[]}


    for i in range(len(file)):
        result["Guid"].append(file[i]["Guid"])
        result["topic"].append(file[i]["topic"])
        result["Q"].append(file[i]["Q"])
        result["First_Image"].append(str(file[i]["First_Image"]))
        result["Second_Image"].append(str(file[i]["Second_Image"]))
        result["image_text"].append(str(file[i]["image_text"]))
        result["A"].append(str(file[i]["A"]))
        result["Qcate"].append(str(file[i]["Qcate"]))
        result["gemini"].append(str(file[i]["gemini"]))
        result["gpt4v"].append(str(file[i]["gpt4v"]))


    return Dataset.from_dict(result)


def transform_single_dialogsumm_file_val(file):
    result = {"Guid":[],"topic":[],"image_text":[],"Q":[], "First_Image": [],"Second_Image": [], "A": [],"Qcate": []}

    # print(file)
    for i in range(len(file)):
        # print()
        result["Guid"].append(file["Guid"][i])
        result["topic"].append(file["topic"][i])
        result["Q"].append(file["Q"][i])
        result["First_Image"].append(str(file["First_Image"][i]))
        result["Second_Image"].append(str(file["Second_Image"][i]))
        result["image_text"].append(str(file["image_text"][i]))
        result["A"].append(str(file["A"][i]))
        result["Qcate"].append(str(file["Qcate"][i]))

    return Dataset.from_dict(result)


def transform_test_file(file):
    result = {"Guid":[],"topic":[],"image_text":[],"Q":[], "First_Image": [],"Second_Image": [], "A": [],"Qcate": []}

    # print(file)
    for i in range(len(file)):
        # print()
        result["Guid"].append(file["Guid"][i])
        result["topic"].append(file["topic"][i])
        result["Q"].append(file["Q"][i])
        result["First_Image"].append(str(file["First_Image"][i]))
        result["Second_Image"].append(str(file["Second_Image"][i]))
        result["image_text"].append(str(file["image_text"][i]))
        result["A"].append(str(file["A"][i]))
        result["Qcate"].append(str(file["Qcate"][i]))


    return Dataset.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(train,validation,test):
    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_single_dialogsumm_file(test)
    # return DatasetDict({"train":train,"validation":validation})
    return DatasetDict({"train":train,"validation":validation, "test": test})

max_input_length = 1024 #265210 #1024 #264192 #2600 #4600    #256
num_epochs = 3
path = "Parent Folder Path"
model = "BLMD_NERNOppo"
config = model+"_WEBQA_xmlcln_Clip512_inTok"+str(max_input_length)
filename_model= config+"_ep_"+str(num_epochs)
print(filename_model)
MODEL_PATH = path+"/Model Path/"+filename_model

is_cuda = torch.cuda.is_available()

import pickle

############ WEBQA ############

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
import pickle

###########################################################################
dataset_val = pd.read_pickle(path+'dataset_split/dataset_val_gem_gpt4.pkl')

dataset_train = pd.read_pickle(path+'dataset_split/dataset_train_gem_gpt4.pkl')
dataset_test = pd.read_pickle(path+'dataset_split/dataset_test_gem_gpt4.pkl')


###########################################################
raw_datasets = transform_dialogsumm_to_huggingface_dataset(dataset_train,dataset_val,dataset_test)
###########################################################


model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

model = get_peft_model(model, peft_config)

print(model.print_trainable_parameters())

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model.config.pad_token_id = tokenizer.eos_token_id

max_target_length = 1024
import ast
rag_output_path = path+"rag_docs/"
video_embed_path = path+"jpgs_clip/"

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

test_vid_embed =[]

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
    for (topic, Qcate, image_text, First_Image, Second_Image) in zip(examples["topic"], examples["Qcate"], examples["image_text"], examples["First_Image"], examples["Second_Image"]):

            try:
                with open(rag_output_path+str(First_Image)+".txt", "r", encoding="UTF-8") as f1:

                    text_1 = f1.read()

                f1.close()
               
            except Exception as e:
                
                text_1 = "NA"
                # print(text_1)
            try:
                with open(rag_output_path+str(Second_Image)+".txt", "r", encoding="UTF-8") as f2:

                    text_2 = f2.read()
                f2.close()
        

            except Exception as e:
            
                text_2 = "NA"

            rag=str("<First Image>"+text_1+"</First Image>"+"<Second Image>"+text_2+"</Second Image>")
        
            # inputs.append(str("Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+str("<First Image>"+text_1+"</First Image>"+"<Second Image>"+text_2+"</Second Image>"))
            inputs.append(str("Task: Question Generation Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+rag)
            inputs_decoder_2.append(str("Task: Multihop Question Generation"+"Topic: "+str(topic)+" "+str(image_text))+rag)
            inputs_decoder_3.append(str("Task: Multihop Question Generation"+"Topic: "+str(topic)+" "+str(image_text))+rag)
            image1.append(str(First_Image))
            image2.append(str(Second_Image))
            rag_out.append(rag)
            try:
                
                img1 = torch.load(video_embed_path+str(First_Image)+".pt").float()
              
            except Exception as e:
               
                img1= torch.zeros(1,512)

            try:
                img2 = torch.load(video_embed_path+str(Second_Image)+".pt").float()
                # print(img2.size)
            except Exception as e:
                # print("img 2 video rmbedd:", e)
                img2= torch.zeros(1,512)
            npy_fileload = torch.cat((img1,img2), dim = 1)

            compressed_vid.append(npy_fileload)



    test_vid_embed.append(compressed_vid)
    vanilla_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    gpt4_inputs = tokenizer(inputs_decoder_2, max_length=max_input_length, truncation=True, padding="max_length")
    gemini_inputs = tokenizer(inputs_decoder_3, max_length=max_input_length, truncation=True, padding="max_length")

    model_inputs = {
        "vanilla_inputs_input_ids": vanilla_inputs["input_ids"],
        "vanilla_inputs_attention_mask": vanilla_inputs["attention_mask"],
        "gpt4_inputs_input_ids": gpt4_inputs["input_ids"],
        "gpt4_inputs_attention_mask": gpt4_inputs["attention_mask"],
        "gemini_inputs_input_ids": gemini_inputs["input_ids"],
        "gemini_inputs_attention_mask": gemini_inputs["attention_mask"],
      
    }

    ################################
    model_inputs["video_embedd"] = compressed_vid
    
    label_input =[]
    gQs_label_input = []

    gpt4Qs_label_input = []
    for qs in examples["Q"]:
        label_input.append(qs)

    for gQs in examples["gemini"]:
        gQs_label_input.append(gQs)

    for gptQs in examples["gpt4v"]:
        gpt4Qs_label_input.append(gptQs)
        # print("gQs_label_input:",gQs_label_input)

    # print(label_input)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():

        # print("Traget Length: "len(label_input))
        # print(examples["question"])
        labels = tokenizer(label_input, max_length=max_target_length, truncation=True)

        labels2 = tokenizer(gQs_label_input, max_length=max_target_length, truncation=True)

        labels_RAG = tokenizer(rag_out, max_length=max_input_length, truncation=True)
        # print(examples["gemini"])
        labels3 = tokenizer(gpt4Qs_label_input, max_length=max_target_length, truncation=True)
        image_1 = tokenizer(image1, max_length=max_target_length, truncation=True)
        image_2 = tokenizer(image2, max_length=max_target_length, truncation=True)


    model_inputs["gemini"] = labels2["input_ids"]
    model_inputs["rag"] = labels_RAG["input_ids"]
    model_inputs["gpt4v"] = labels3["input_ids"]

    model_inputs["image_id_1"] = image_1["input_ids"]
    model_inputs["image_id_2"] = image_2["input_ids"]

        # print(labels)
    model_inputs["labels"] = labels["input_ids"]
    # print("model_inputs: ",model_inputs)
    return model_inputs


def preprocess_function_val(examples):
    inputs = []
    compressed_vid = []
    model_inputs = {}
    input_embedd_feat = []
    rag_out = []
    for (topic, Qcate, image_text, First_Image, Second_Image) in zip(examples["topic"], examples["Qcate"], examples["image_text"], examples["First_Image"], examples["Second_Image"]):

            try:
                with open(rag_output_path+str(First_Image)+".txt", "r", encoding="UTF-8") as f1:

                    text_1 = f1.read()

                f1.close()
            except Exception as e:
            
                text_1 = "NA"
               
            try:
                with open(rag_output_path+str(Second_Image)+".txt", "r", encoding="UTF-8") as f2:

                    text_2 = f2.read()
                f2.close()
               

            except Exception as e:
            
                text_2 = "NA"

            rag=str("<First Image>"+text_1+"</First Image>"+"<Second Image>"+text_2+"</Second Image>")
            inputs.append(str("Task: Multihop Question Generation Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+rag)

            rag_out.append(rag)
            try:
     
                img1 = torch.load(video_embed_path+str(First_Image)+".pt").float()
                
            except Exception as e:
            
                img1= torch.zeros(1,512)

            try:
                img2 = torch.load(video_embed_path+str(Second_Image)+".pt").float()
                
            except Exception as e:
               
                img2= torch.zeros(1,512)
            npy_fileload = torch.cat((img1,img2), dim = 1)

            compressed_vid.append(npy_fileload)

    test_vid_embed.append(compressed_vid)
    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation = True)
  
    ################################
    model_inputs["video_embedd"] = compressed_vid
    label_input = []

    for qs in examples["Q"]:
        label_input.append(qs)

    with tokenizer.as_target_tokenizer():

        labels = tokenizer(label_input, max_length = max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def preprocess_function_test(examples):
    inputs = []
    compressed_vid = []
    model_inputs = {}
    input_embedd_feat = []
    rag_out = []
    for (topic, Qcate, image_text, First_Image, Second_Image) in zip(examples["topic"], examples["Qcate"], examples["image_text"], examples["First_Image"], examples["Second_Image"]):

            try:
                with open(rag_output_path+str(First_Image)+".txt", "r", encoding="UTF-8") as f1:

                    text_1 = f1.read()

                f1.close()
             
            except Exception as e:
            
                text_1 = "NA"
              
            try:
                with open(rag_output_path+str(Second_Image)+".txt", "r", encoding="UTF-8") as f2:

                    text_2 = f2.read()
                f2.close()
            except Exception as e:

                text_2 = "NA"

            rag=str("<First Image>"+text_1+"</First Image>"+"<Second Image>"+text_2+"</Second Image>")
            inputs.append(str("Task: Multihop Question Generation Topic: "+str(topic)+" Queston_Type: "+str(Qcate)+" "+str(image_text))+rag)

            rag_out.append(rag)
            try:
                img1 = torch.load(video_embed_path+str(First_Image)+".pt").float()
              
            except Exception as e:
            
                img1= torch.zeros(1,512)

            try:
                img2 = torch.load(video_embed_path+str(Second_Image)+".pt").float()
               
            except Exception as e:
           
                img2= torch.zeros(1,512)
            npy_fileload = torch.cat((img1,img2), dim = 1)

            compressed_vid.append(npy_fileload)

    test_vid_embed.append(compressed_vid)
    ################################
    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation = True)
    
    ################################
    model_inputs["video_embedd"] = compressed_vid
    label_input = []

    for qs in examples["Q"]:
        label_input.append(qs)

    with tokenizer.as_target_tokenizer():

        labels = tokenizer(label_input, max_length = max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
 
    return model_inputs



raw_datasets = transform_dialogsumm_to_huggingface_dataset(dataset_train,dataset_val,dataset_test)

#####################################################
tokenized_datasets = raw_datasets["train"].map(preprocess_function, batched=True)
tokenized_datasets_test = raw_datasets["test"].map(preprocess_function_test, batched=True)
tokenized_datasets_val1 = raw_datasets["validation"].map(preprocess_function_val, batched=True)

#####################################################
import pickle

import time

label_names=['labels','gemini','gpt4v',"rag"]
batch_size = 1
args = Seq2SeqTrainingArguments(
    config+"rog_ep_"+str(num_epochs),
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    save_strategy="epoch",
    metric_for_best_model="eval_rouge1",
    greater_is_better=True,
    seed=42,
    generation_max_length=max_target_length,
    logging_strategy = "epoch",
    remove_unused_columns=False,
    label_names = label_names,
    gradient_accumulation_steps=4,
    fp16=True
)


def collate_fn(batch):
    # inputs = [item['input_ids'] for item in batch]
    flag = 0
    try:
        inputs_vanilla = [item['vanilla_inputs_input_ids'] for item in batch]
        inputs_gpt4 = [item['gpt4_inputs_input_ids'] for item in batch]
        inputs_gemini = [item['gemini_inputs_input_ids'] for item in batch]
        mask_vanilla = [item['vanilla_inputs_attention_mask'] for item in batch]
        mask_gpt4 = [item['gpt4_inputs_attention_mask'] for item in batch]
        mask_gemini = [item['gemini_inputs_attention_mask'] for item in batch]
        image_1 = [item['image_id_1'] for item in batch]
        image_2 = [item['image_id_2'] for item in batch]

        flag = 0
    except Exception as e:
        # print("Front end 1:", e)
        # inputs = [item['input_ids'] for item in batch]
        # mask = [item['attention_mask'] for item in batch]
        inputs_vanilla = [item['input_ids'] for item in batch]
        inputs_gpt4 = [item['input_ids'] for item in batch]
        inputs_gemini = [item['input_ids'] for item in batch]
        mask_vanilla =  [item['attention_mask'] for item in batch]
        mask_gpt4 = [item['attention_mask'] for item in batch]
        mask_gemini =  [item['attention_mask'] for item in batch]
        image_1 = [item['labels'] for item in batch]


        image_2 = [item['labels'] for item in batch]
        flag = 1
    # inputs = [item['inputs_embeds'] for item in batch]
    labels = [item['labels'] for item in batch]

    try:
        labels_gemini = [item['gemini'] for item in batch]
        labels_rag = [item['rag'] for item in batch]
        labels_gpt = [item['gpt4v'] for item in batch]
        flag = 0
    except Exception as e:
        # print("Front end 2:", e)
        flag = 1
        labels_gemini = [item['labels'] for item in batch]

        labels_rag = [item['labels'] for item in batch]
        labels_gpt = [item['labels'] for item in batch]




    #####################################################
    video_embeds = [item['video_embedd'] for item in batch]
    #####################################################

    inputs_text_vanilla = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in inputs_vanilla], batch_first=True, padding_value=0)
    inputs_text_gpt4 = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in inputs_gpt4], batch_first=True, padding_value=0)
    inputs_text_gemini = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in inputs_gemini], batch_first=True, padding_value=0)
    labels_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in labels], batch_first=True, padding_value=0)

    labels_text_gem = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in labels_gemini], batch_first=True, padding_value=0)


    labels_text_rag = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in labels_rag], batch_first=True, padding_value=0)

    labels_text_gpt = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in labels_gpt], batch_first=True, padding_value=0)
    attention_mask_vanilla = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in mask_vanilla], batch_first=True, padding_value=0)
    attention_mask_gpt4 = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in mask_gpt4], batch_first=True, padding_value=0)
    attention_mask_gemini = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in mask_gemini], batch_first=True, padding_value=0)


    First_Image = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in image_1], batch_first=True, padding_value=0)


    Second_Image = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in image_2], batch_first=True, padding_value=0)

    #####################################################
    video_embeds_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in video_embeds], batch_first=True, padding_value=0)

    if flag == 1:
        flag = 0
        return {'input_ids': inputs_text_vanilla, 'labels': labels_text, 'video_embedd': video_embeds_padded, 'attention_mask': attention_mask_vanilla}
    else:
        return {'vanilla_inputs_input_ids': inputs_text_vanilla,
        'gpt4_inputs_input_ids': inputs_text_gpt4,
        'gemini_inputs_input_ids': inputs_text_gemini,
        'rag': labels_text_rag,
        'gpt4v': labels_text_gpt,
        'gemini': labels_text_gem, 'labels': labels_text,
        'video_embedd': video_embeds_padded,
        'vanilla_inputs_attention_mask': attention_mask_vanilla,
        'gpt4_inputs_attention_mask': attention_mask_gpt4,
        'gemini_inputs_attention_mask': attention_mask_gemini,
        "image_id_1": First_Image,
        "image_id_2": Second_Image  }


from transformers import DataCollator,DefaultDataCollator
from rouge_score import rouge_scorer
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

data_collator = DefaultDataCollator()

import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions = [pred if pred < tokenizer.vocab_size else tokenizer.unk_token_id for pred in predictions]
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v , 4) for k, v in result.items()}



trainer = CustomTrainer(
    model,
    args,
    train_dataset= tokenized_datasets,
    eval_dataset=tokenized_datasets_val1,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()

list_val=[tokenized_datasets_test]
final_decoded_preds=[]
for val_data in list_val:
    out = trainer.predict(val_data, num_beams=4, max_length=128) #test
    predictions = out.predictions

    # Ensure predictions are in a manageable range
    predictions = np.clip(predictions, a_min=0, a_max=tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]

    final_decoded_preds.append(decoded_preds)




with open(MODEL_PATH+".json", "a") as outfile:
    outfile.write('[')
    for index, item in enumerate(dataset_test):
        dictionary = {"Guid": str(item["Guid"]),"Gold_Question":str(item["Q"]),"Generated_Question":decoded_preds[index]}
        # print(dictionary)

        if index > 0:
            outfile.write(',')
        json.dump(dictionary, outfile)
    outfile.write(']')


print(f"{config+'ep_'+str(num_epochs)}Trined Successfully! Output Saved.")


