print("start")
from transformers import RobertaForSequenceClassification, AdamW, BertConfig, BertTokenizer, BertPreTrainedModel, BertModel, BertForMultipleChoice, RobertaForMultipleChoice, RobertaTokenizer, RobertaModel
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import json
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
import random
from transformers.modeling_utils import PreTrainedModel

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import torchvision.transforms as T
from transformers import ViTModel,ViTFeatureExtractor
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F
print("start")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')    # feature extractor from VIT
#model = ViTModel.from_pretrained("google/vit-large-patch16-224")
import os
#os.environ['CUDA_VISIBLE_DEVICES']= '0,3'
device = torch.device("cuda:2")
def load_model():
    lm_model = torch.load("/data/linqika/xufangzhi/ISAAQ/checkpoints/pretrain_physics+tqa_spanmask_RACE_e2.pth")
    mc_model = RobertaForMultipleChoice.from_pretrained('roberta-large')

    mc_model.roberta.embeddings.load_state_dict(lm_model.roberta.embeddings.state_dict())
    mc_model.roberta.encoder.load_state_dict(lm_model.roberta.encoder.state_dict())

    return mc_model

class VIT_mcan_L(torch.nn.Module):   #多层mcan
    def __init__(self):
        super(VIT_mcan_L, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        
        self.K_linear_L1 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.V_linear_L1 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.Q_linear_L1 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        
        self.K_linear_L2 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.V_linear_L2 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.Q_linear_L2 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        
        self.K_linear_L3 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.V_linear_L3 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.Q_linear_L3 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        
        self.K_linear_L4 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.V_linear_L4 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.Q_linear_L4 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        
        self.K_linear_L5 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.V_linear_L5 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.Q_linear_L5 = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        
        #self.final_text_linear = torch.nn.Linear(1024,1024)
        #self.final_img_linear = torch.nn.Linear(1024,1024)
        #self.final_layernorm = torch.nn.LayerNorm([4,1024])
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[0]  # [batch_size, seq_size, hidden_size]
        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
        

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)   # [batch_size, hidden_size]
        out_vit1 = out_vit1.unsqueeze(1)    # [batch_size, 1, hidden_size]
        
        # Layer 1
        Q_img_L1 = self.Q_linear_L1(out_vit1)
        K_text_L1 = self.K_linear_L1(out_roberta).permute(0,2,1)
        V_text_L1 = self.V_linear_L1(out_roberta)        
        alpha = torch.matmul(Q_img_L1, K_text_L1)/ torch.sqrt(torch.tensor(1024.0))
        alpha = F.softmax(alpha, dim = 2)
        GA_out = torch.matmul(alpha, V_text_L1)  #[batch_size, seq_size, hidden_state]
        
        # Layer 2
        Q_img_L2 = self.Q_linear_L2(GA_out[:,0,:])
        K_text_L2 = self.K_linear_L2(out_roberta).permute(0,2,1)
        V_text_L2 = self.V_linear_L2(out_roberta)        
        alpha = torch.matmul(Q_img_L2, K_text_L2)/ torch.sqrt(torch.tensor(1024.0))
        alpha = F.softmax(alpha, dim = 2)
        GA_out = torch.matmul(alpha, V_text_L2)  #[batch_size, seq_size, hidden_state]
        
        
        # Layer 3
        Q_img_L3 = self.Q_linear_L3(GA_out[:,0,:])
        K_text_L3 = self.K_linear_L3(out_roberta).permute(0,2,1)
        V_text_L3 = self.V_linear_L3(out_roberta)        
        alpha = torch.matmul(Q_img_L3, K_text_L3)/ torch.sqrt(torch.tensor(1024.0))
        alpha = F.softmax(alpha, dim = 2)
        GA_out = torch.matmul(alpha, V_text_L3)  #[batch_size, seq_size, hidden_state]
        
        
        # Layer 4
        Q_img_L4 = self.Q_linear_L4(GA_out[:,0,:])
        K_text_L4 = self.K_linear_L4(out_roberta).permute(0,2,1)
        V_text_L4 = self.V_linear_L4(out_roberta)        
        alpha = torch.matmul(Q_img_L4, K_text_L4)/ torch.sqrt(torch.tensor(1024.0))
        alpha = F.softmax(alpha, dim = 2)
        GA_out = torch.matmul(alpha, V_text_L4)  #[batch_size, seq_size, hidden_state]
        
        
        # Layer 5
        Q_img_L5 = self.Q_linear_L5(GA_out[:,0,:])
        K_text_L5 = self.K_linear_L5(out_roberta).permute(0,2,1)
        V_text_L5 = self.V_linear_L5(out_roberta)        
        alpha = torch.matmul(Q_img_L5, K_text_L5)/ torch.sqrt(torch.tensor(1024.0))
        alpha = F.softmax(alpha, dim = 2)
        GA_out = torch.matmul(alpha, V_text_L5)  #[batch_size, seq_size, hidden_state]
        
        final_out = GA_out[:,0,:]
        #final_out = self.final_layernorm(self.final_text_linear(out_roberta[:,0,:]) + self.final_img_linear(GA_out[:,0,:]))

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
        
class VIT_mcan(torch.nn.Module):
    def __init__(self):
        super(VIT_mcan, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        
        self.K_linear = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.V_linear = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())
        self.Q_linear = torch.nn.Sequential(torch.nn.Linear(1024,1024),torch.nn.Tanh())

        self.final_text_linear = torch.nn.Linear(1024,1024)
        self.final_img_linear = torch.nn.Linear(1024,1024)
        self.final_layernorm = torch.nn.LayerNorm([4,1024])
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[0]  # [batch_size, seq_size, hidden_size]
        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
        

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)   # [batch_size, hidden_size]
        out_vit1 = out_vit1.unsqueeze(1)    # [batch_size, 1, hidden_size]
        
        
        Q_img = self.Q_linear(out_vit1)
        K_text = self.K_linear(out_roberta).permute(0,2,1)
        V_text = self.V_linear(out_roberta)        
        
        alpha = torch.matmul(Q_img, K_text)/ torch.sqrt(torch.tensor(1024.0))
        
        alpha = F.softmax(alpha, dim = 2)
        GA_out = torch.matmul(alpha, V_text)  #[batch_size, seq_size, hidden_state]

        #final_out = GA_out[:,0,:]

        final_out = self.final_layernorm(self.final_text_linear(out_roberta[:,0,:]) + self.final_img_linear(GA_out[:,0,:]))

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
        
class VIT_bd_v2_image_caption(torch.nn.Module):
    def __init__(self):
        super(VIT_bd_v2_image_caption, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        self.att_ques = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        self.att_bd = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
    

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)

        out_att_ques = torch.cat((out_roberta, out_vit1),1)
        out_att_ques = self.att_ques(out_att_ques)
        out_att_ques = out_att_ques*out_vit1
        ques_part_out = torch.sum(out_att_ques, dim=0)
        
        ques_part_out = out_roberta * ques_part_out
        
        out_vit2 = torch.stack(img_b2, dim=0).to(device)
        out_vit2 = self.model(out_vit2).last_hidden_state[:,0,:].to(device)
        out_vit2 = out_vit2.view(-1,1024)

        out_att_bd = torch.cat((out_vit2, ques_part_out),1)
        out_att_bd = self.att_bd(out_att_bd)    #modify
        out_att_bd = out_att_bd*out_vit2  
        bd_part_out = torch.sum(out_att_bd, dim=0)


        final_out = bd_part_out * ques_part_out 

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
class VIT_bd_v5(torch.nn.Module):
    def __init__(self):
        super(VIT_bd_v5, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        self.att_ques = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        self.att_bd = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
    

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)

        out_att_ques = torch.cat((out_roberta, out_vit1),1)
        out_att_ques = self.att_ques(out_att_ques)
        out_att_ques = out_att_ques*out_vit1
        ques_part_out = torch.sum(out_att_ques, dim=0)
        
        ques_part_out = out_roberta * ques_part_out
        
        out_vit2 = torch.stack(img_b2, dim=0).to(device)
        out_vit2 = self.model(out_vit2).last_hidden_state[:,0,:].to(device)
        out_vit2 = out_vit2.view(-1,1024)

        out_att_bd = torch.cat((out_vit2, ques_part_out),1)
        out_att_bd = self.att_bd(out_att_bd)    #modify
        out_att_bd = out_att_bd*out_vit2  
        bd_part_out = torch.sum(out_att_bd, dim=0)


        final_out = bd_part_out * out_roberta

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
        
class VIT_bd_v4(torch.nn.Module):
    def __init__(self):
        super(VIT_bd_v4, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        self.att_ques = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        self.att_bd = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
    

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)

        out_att_ques = torch.cat((out_roberta, out_vit1),1)
        out_att_ques = self.att_ques(out_att_ques)
        out_att_ques = out_att_ques*out_vit1
        ques_part_out = torch.sum(out_att_ques, dim=0)
        
        ques_part_out = out_roberta * ques_part_out
        
        out_vit2 = torch.stack(img_b2, dim=0).to(device)
        out_vit2 = self.model(out_vit2).last_hidden_state[:,0,:].to(device)
        out_vit2 = out_vit2.view(-1,1024)

        out_att_bd = torch.cat((out_vit2, ques_part_out),1)
        out_att_bd = self.att_bd(out_att_bd)    #modify
        out_att_bd = out_att_bd*out_vit2  
        bd_part_out = torch.sum(out_att_bd, dim=0)


        final_out = bd_part_out * ques_part_out * out_roberta

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
        
class VIT_bd_shareweight(torch.nn.Module):
    def __init__(self):
        super(VIT_bd_shareweight, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        self.att_share = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        self.att_ques = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        self.att_bd = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
    

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)

        out_att_ques = torch.cat((out_roberta, out_vit1),1)
        out_att_ques = self.att_ques(out_att_ques)
        out_att_ques = out_att_ques*out_vit1
        ques_part_out = torch.sum(out_att_ques, dim=0)
        ques_part_out = out_roberta * ques_part_out
        

        out_vit2 = torch.stack(img_b2, dim=0).to(device)
        out_vit2 = self.model(out_vit2).last_hidden_state[:,0,:].to(device)
        out_vit2 = out_vit2.view(-1,1024)

        out_att_bd = torch.cat((out_roberta, out_vit2),1)
        out_att_bd = self.att_bd(out_att_bd)    #modify
        out_att_bd = out_att_bd*out_vit2  
        bd_part_out = torch.sum(out_att_bd, dim=0)
        bd_part_out = out_roberta * bd_part_out

        final_out = bd_part_out * ques_part_out

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
        
class VIT_bd_v2(torch.nn.Module):
    def __init__(self):
        super(VIT_bd_v2, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        self.att_ques = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        self.att_bd = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
    

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)

        out_att_ques = torch.cat((out_roberta, out_vit1),1)
        out_att_ques = self.att_ques(out_att_ques)
        out_att_ques = out_att_ques*out_vit1
        ques_part_out = torch.sum(out_att_ques, dim=0)
        
        ques_part_out = out_roberta * ques_part_out
        
        out_vit2 = torch.stack(img_b2, dim=0).to(device)
        out_vit2 = self.model(out_vit2).last_hidden_state[:,0,:].to(device)
        out_vit2 = out_vit2.view(-1,1024)

        out_att_bd = torch.cat((out_vit2, ques_part_out),1)
        out_att_bd = self.att_bd(out_att_bd)    #modify
        out_att_bd = out_att_bd*out_vit2  
        bd_part_out = torch.sum(out_att_bd, dim=0)


        final_out = bd_part_out * ques_part_out 

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
        
class VIT_bd(torch.nn.Module):
    def __init__(self):
        super(VIT_bd, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        self.att = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images1=None, images2=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b1 = []    #question image
        for img_q in images1:
            for img_file in img_q:
                img_b1.append(vit_get_images(img_file)["pixel_values"].squeeze())

        img_b2 = []    #background image
        for img_q in images2:
            for img_file in img_q:
                img_b2.append(vit_get_images(img_file)["pixel_values"].squeeze())
    

        out_vit1 = torch.stack(img_b1, dim=0).to(device)
        out_vit1 = self.model(out_vit1).last_hidden_state[:,0,:].to(device)
        out_vit1 = out_vit1.view(-1,1024)

        ques_part_out = out_roberta * out_vit1
        
        out_vit2 = torch.stack(img_b2, dim=0).to(device)
        out_vit2 = self.model(out_vit2).last_hidden_state[:,0,:].to(device)
        out_vit2 = out_vit2.view(-1,1024)

        out_att = torch.cat((out_vit2, ques_part_out),1)
        out_att = self.att(out_att)
        out_att = out_att*out_vit2  
        bd_part_out = torch.sum(out_att, dim=0)

        final_out = bd_part_out * ques_part_out 

        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
          
class VIT(torch.nn.Module):
    def __init__(self):
        super(VIT, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        self.model = ViTModel.from_pretrained("google/vit-large-patch16-224")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images=None, labels=None):
        num_choices = input_ids.shape[1]
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]

        img_b = []
        for img_q in images:
            for img_file in img_q:
                img_b.append(vit_get_images(img_file)["pixel_values"].squeeze())

        out_vit = torch.stack(img_b, dim=0).cuda()

        out_vit = self.model(out_vit).last_hidden_state[:,0,:].cuda()
        out_vit = out_vit.view(-1,1024)

        final_out = out_roberta * out_vit
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs

class ResnetRoberta(torch.nn.Module):
    def __init__(self):
        super(ResnetRoberta, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        
        self.resnet=models.resnet101(pretrained=True)
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000,1024))
        
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images=None, labels=None):
        num_choices = input_ids.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]
        
        img_b = []
        for img_q in images:
            for img_file in img_q:
                img_b.append(get_images(img_file))
        out_resnet = torch.stack(img_b, dim=0)
        out_resnet = self.resnet(out_resnet)
        out_resnet = self.feats(out_resnet)
        out_resnet = out_resnet.view(-1,1024)

        final_out = out_roberta * out_resnet
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
    
class ResnetRobertaBU(torch.nn.Module):
    def __init__(self):
        super(ResnetRobertaBU, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        
        self.resnet = models.resnet101(pretrained=True)
        
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000,1024))
        self.feats2 = torch.nn.Sequential(torch.nn.LayerNorm(1024, eps=1e-12))
        
        self.boxes = torch.nn.Sequential(torch.nn.Linear(4,1024),torch.nn.LayerNorm(1024, eps=1e-12))
        
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images=None,coords=None,labels=None):
        num_choices = input_ids.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]
        
        img_b = []
        coord_b = []
        for img_q, coord_q in zip(images,coords):
            for img_file, coord in zip(img_q,coord_q):
                img_b.append(img_file)
                coord_b.append(coord)
        
        roi_b = []
        for image, coord, roberta_b in zip(img_b,coord_b, out_roberta):
            img_v = get_rois(image, coord[:32])
            coord_v = torch.tensor(coord[:32]).cuda()
            out_boxes = self.boxes(coord_v)
            out_resnet = self.resnet(img_v)
            out_resnet = self.feats(out_resnet)
            out_resnet = self.feats2(out_resnet)
            out_resnet = out_resnet.view(-1,1024)
            out_roi = (out_resnet + out_boxes)/2
            out_roi = torch.sum(out_roi, dim=0)
            roi_b.append(out_roi)
        out_visual = torch.stack(roi_b, dim=0)
        
        final_out = out_roberta * out_visual
        
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
    
class ResnetRobertaBUTD(torch.nn.Module):
    def __init__(self):
        super(ResnetRobertaBUTD, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        
        self.resnet = models.resnet101(pretrained=True)
        
        self.feats = torch.nn.Sequential(torch.nn.Linear(1000,1024))
        self.feats2 = torch.nn.Sequential(torch.nn.LayerNorm(1024, eps=1e-12))
        
        self.boxes = torch.nn.Sequential(torch.nn.Linear(4,1024),torch.nn.LayerNorm(1024, eps=1e-12))
        
        self.att = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Tanh())
        
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(1024, 1)
        
    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,images=None,coords=None,labels=None):
        num_choices = input_ids.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask)[1]
        
        img_b = []
        coord_b = []

        for img_q, coord_q in zip(images,coords):
            for img_file, coord in zip(img_q,coord_q):
                img_b.append(img_file)
                coord_b.append(coord)
        
        roi_b = []
        for image, coord, roberta_b in zip(img_b,coord_b, out_roberta):
            img_v = get_rois(image, coord[:32])
            coord_v = torch.tensor(coord[:32]).cuda()
            out_boxes = self.boxes(coord_v)
            out_resnet = self.resnet(img_v)
            out_resnet = self.feats(out_resnet)
            out_resnet = self.feats2(out_resnet)
            out_resnet = out_resnet.view(-1,1024)
            out_roi = (out_resnet + out_boxes)/2
            
            n_rois = np.shape(out_roi)[0]
            
            out_att = torch.cat((out_roi, roberta_b.repeat(n_rois,1)),1)
            out_att = self.att(out_att)
            out_att = out_att*out_roi
            out_att = torch.sum(out_att, dim=0)
            roi_b.append(out_att)
        out_visual = torch.stack(roi_b, dim=0)
        
        final_out = out_roberta * out_visual
        
        pooled_output = self.dropout(final_out)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs
        
        
        

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    #return np.sum(pred_flat == labels_flat) / len(labels_flat)
    return np.sum(pred_flat == labels_flat), np.sum(pred_flat != labels_flat)


def vit_get_images(img_path):
    im = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im)
    return feature_extractor(images=img)
    
    
def get_images(img_path):
    image = Image.open(img_path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.array(image)[:,:,:3]
    image = torch.tensor(image).type(torch.FloatTensor).permute(2,0,1).cuda()
    return image

def vit_get_rois(img_path, vectors):
    image = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    rois = []
    for vector in vectors:
        image = image.crop(vector)
        img = transform(image)
        roi_image = feature_extractor(images=img)["pixel_values"].squeeze()
        rois.append(roi_image)
    rois = torch.stack(rois, dim=0)
    return rois
    
def get_rois(img_path, vectors):
    image = Image.open(img_path)
    rois = []
    for vector in vectors:
        roi_image = image.crop(vector)
        roi_image = roi_image.resize((224, 224), Image.ANTIALIAS)
        roi_image = np.array(roi_image)
        roi_image = torch.tensor(roi_image).type(torch.FloatTensor).permute(2,0,1).cuda()
        rois.append(roi_image)
    rois = torch.stack(rois, dim=0)
    return rois



def get_choice_encoded(text, question, answer, max_len, tokenizer):
    if text != "":
        first_part = text
        second_part = question + " " + answer
        encoded = tokenizer.encode_plus(first_part, second_part, max_length=max_len, pad_to_max_length=True)
    else:
        encoded = tokenizer.encode_plus(question + " " + answer, max_length=max_len, pad_to_max_length=True)
    input_ids = encoded["input_ids"]
    att_mask = encoded["attention_mask"]
    return input_ids, att_mask

def get_data_tf(split, retrieval_solver, tokenizer, max_len):
    input_ids_list=[]
    att_mask_list=[]
    labels_list=[]
    with open("jsons/tqa_tf_concat.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["sentence_"+retrieval_solver]
        encoded = tokenizer.encode_plus(text, question, max_length=max_len, pad_to_max_length=True)
        input_ids = encoded["input_ids"]
        att_mask = encoded["attention_mask"]
        label = 0
        if doc["correct_answer"] == "a":
            label = 1
        input_ids_list.append(input_ids)
        att_mask_list.append(att_mask)
        labels_list.append(label)
    return [input_ids_list, att_mask_list, labels_list]

def get_data_ndq(dataset_name, split, retrieval_solver, tokenizer, max_len):
    input_ids_list=[]
    att_mask_list=[]
    labels_list=[]
    cont = 0
    #with open("jsons/tqa_"+dataset_name+"_concat.json", "r", encoding="utf-8", errors="surrogatepass") as file:
    with open("jsons/tqa_"+dataset_name+".json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_"+retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q = []
        att_mask_q = []
        if dataset_name == "ndq":
            counter = 7
        if dataset_name == "dq":
            counter = 4
        for count_i in range(counter):
            try:
                answer = answers[count_i]
            except:
                answer = ""
            input_ids_aux, att_mask_aux = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(input_ids_aux)
            att_mask_q.append(att_mask_aux)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        label = list(doc["answers"].keys()).index(doc["correct_answer"])
        labels_list.append(label)
    return [input_ids_list, att_mask_list, labels_list]

def process_data_ndq(raw_data, batch_size, split):
    input_ids_list, att_mask_list, labels_list = raw_data
    inputs = torch.tensor(input_ids_list)
    masks = torch.tensor(att_mask_list)
    labels = torch.tensor(labels_list)
    
    if split=="train":
        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        data = TensorDataset(inputs, masks, labels)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    return dataloader

def get_data_dq(split, retrieval_solver, tokenizer, max_len):
    input_ids_list=[]
    att_mask_list=[]
    images_list = []
    coords_list = []
    labels_list=[]
    cont = 0
    with open("jsons/tqa_dq.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_"+retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q = []
        att_mask_q = []
        images_q = []
        coords_q = []
        for count_i in range(4):
            try:
                answer = answers[count_i]
            except:
                answer = ""
            input_ids_aux, att_mask_aux = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(input_ids_aux)
            att_mask_q.append(att_mask_aux)
            images_q.append(doc["image_path"])
            coord = [c[:4] for c in doc["coords"]]
            coords_q.append(coord)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        images_list.append(images_q)
        coords_list.append(coords_q)
        label = list(doc["answers"].keys()).index(doc["correct_answer"])
        labels_list.append(label)
    return [input_ids_list, att_mask_list, images_list, coords_list, labels_list]

def get_data_dq_bd(split, retrieval_solver, tokenizer, max_len):
    input_ids_list=[]
    att_mask_list=[]
    images1_list = []
    images2_list = []
    coords_list = []
    labels_list=[]
    cont = 0
    with open("jsons/tqa_dq_bd.json", "r", encoding="utf-8", errors="surrogatepass") as file:
        dataset = json.load(file)
    dataset = [doc for doc in dataset if doc["split"] == split]
    for doc in tqdm(dataset):
        question = doc["question"]
        text = doc["paragraph_"+retrieval_solver]
        answers = list(doc["answers"].values())
        input_ids_q = []
        att_mask_q = []
        images1_q = []
        images2_q = []
        coords_q = []
        for count_i in range(4):
            try:
                answer = answers[count_i]
            except:
                answer = ""
            input_ids_aux, att_mask_aux = get_choice_encoded(text, question, answer, max_len, tokenizer)
            input_ids_q.append(input_ids_aux)
            att_mask_q.append(att_mask_aux)
            images1_q.append(doc["image_path"])
            images2_q.append(doc["context_image_path"])
            coord = [c[:4] for c in doc["coords"]]
            coords_q.append(coord)
        input_ids_list.append(input_ids_q)
        att_mask_list.append(att_mask_q)
        images1_list.append(images1_q)
        images2_list.append(images2_q)
        coords_list.append(coords_q)
        label = list(doc["answers"].keys()).index(doc["correct_answer"])
        labels_list.append(label)
    return [input_ids_list, att_mask_list, images1_list, images2_list, coords_list, labels_list]

def training_tf(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device, save_model=False):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        pbar = tqdm(train_dataloader)
        for batch in pbar:  
            model.train()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            #print(batch[0].size(), batch[1].size(), batch[2].size())
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss,logits = outputs[0], outputs[1]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points/(total_points+total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/tf_6-10_"+retrieval_solver+"_e"+str(epoch_i+1)+".pth")
        
        validation_tf(model, val_dataloader, device)
        
    print("")
    print("Training complete!")
        
def validation_tf(model, val_dataloader, device):

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    total_points = 0
    total_errors = 0
    val_loss_list = []
    final_res = []

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    for batch in tqdm(val_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss, logits = outputs[0], outputs[1]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        for l in logits:
            final_res.append(l)
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        points, errors = flat_accuracy(logits, label_ids)
        total_points = total_points + points
        total_errors = total_errors + errors

        val_loss_list.append(loss.item())

    val_acc = total_points/(total_points+total_errors)
    val_loss = np.mean(val_loss_list)

    print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
    
    return final_res
    
def training_ndq(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device, save_model, dataset_name):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        pbar = tqdm(train_dataloader)
        for batch in pbar:  
            model.train()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
            loss,logits = outputs[0],outputs[1]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points/(total_points+total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/diagram_checkpoints/dmc_"+dataset_name+"_roberta_"+retrieval_solver+"_e"+str(epoch_i+1)+".pth")
        
        validation_ndq(model, val_dataloader, device)
        
    print("")
    print("Training complete!")
        
    
def validation_ndq(model, val_dataloader, device):

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    total_points = 0
    total_errors = 0
    val_loss_list = []
    final_res = []

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    for batch in tqdm(val_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss, logits = outputs[0],outputs[1]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        for l in logits:
            final_res.append(l)
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        points, errors = flat_accuracy(logits, label_ids)
        total_points = total_points + points
        total_errors = total_errors + errors

        val_loss_list.append(loss.item())

    val_acc = total_points/(total_points+total_errors)
    val_loss = np.mean(val_loss_list)

    print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
    
    return final_res

def training_dq(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device, save_model):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        input_ids_list, att_mask_list, images_list, coords_list, labels_list = raw_data_train

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        dataset_ids = list(range(len(labels_list)))
        random.shuffle(dataset_ids)
        batched_ids = [dataset_ids[k:k+batch_size] for k in range(0, len(dataset_ids), batch_size)]
        pbar = tqdm(batched_ids)
        for batch_ids in pbar:  
            model.train()
            b_input_ids = torch.tensor([x for y,x in enumerate(input_ids_list) if y in batch_ids]).to(device)
            b_input_mask = torch.tensor([x for y,x in enumerate(att_mask_list) if y in batch_ids]).to(device)
            b_input_images = [x for y,x in enumerate(images_list) if y in batch_ids]
            b_input_coords = [x for y,x in enumerate(coords_list) if y in batch_ids]
            b_labels = torch.tensor([x for y,x in enumerate(labels_list) if y in batch_ids]).to(device)
            #outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            #outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, coords=b_input_coords, labels=b_labels)
            outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, labels=b_labels)
            
            loss,logits = outputs[0], outputs[1]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points/(total_points+total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/diagram_checkpoints/dmc_dq_VIT_VQA_AI2De8_"+retrieval_solver+"_e"+str(epoch_i+1)+".pth")
        
        validation_dq(model, raw_data_val, batch_size, device)
        
    print("")
    print("Training complete!")
    
def validation_dq(model, raw_data_val, batch_size, device):

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")
    
    input_ids_list, att_mask_list, images_list, coords_list, labels_list = raw_data_val

    total_points = 0
    total_errors = 0
    val_loss_list = []
    final_res = []

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    dataset_ids = list(range(len(labels_list)))
    batched_ids = [dataset_ids[k:k+batch_size] for k in range(0, len(dataset_ids), batch_size)]
    pbar = tqdm(batched_ids)
    for batch_ids in pbar:
        # Unpack the inputs from our dataloader
        b_input_ids = torch.tensor([x for y,x in enumerate(input_ids_list) if y in batch_ids]).to(device)
        b_input_mask = torch.tensor([x for y,x in enumerate(att_mask_list) if y in batch_ids]).to(device)
        b_input_images = [x for y,x in enumerate(images_list) if y in batch_ids]
        b_input_coords = [x for y,x in enumerate(coords_list) if y in batch_ids]
        b_labels = torch.tensor([x for y,x in enumerate(labels_list) if y in batch_ids]).to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            #outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)        
            #outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, coords=b_input_coords, labels=b_labels)
             outputs = model(b_input_ids, attention_mask=b_input_mask, images=b_input_images, labels=b_labels)

        loss, logits = outputs[0], outputs[1]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        for l in logits:
            final_res.append(l)
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        points, errors = flat_accuracy(logits, label_ids)
        total_points = total_points + points
        total_errors = total_errors + errors

        val_loss_list.append(loss.item())

    val_acc = total_points/(total_points+total_errors)
    val_loss = np.mean(val_loss_list)

    print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
    
    return final_res

def training_dq_bd(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device, save_model):
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        input_ids_list, att_mask_list, images1_list, images2_list, coords_list, labels_list = raw_data_train

        # Reset the total loss for this epoch.
        total_points = 0
        total_errors = 0
        train_loss_list = []

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # For each batch of training data...
        dataset_ids = list(range(len(labels_list)))
        random.shuffle(dataset_ids)
        batched_ids = [dataset_ids[k:k+batch_size] for k in range(0, len(dataset_ids), batch_size)]
        pbar = tqdm(batched_ids)
        for batch_ids in pbar:  
            model.train()
            b_input_ids = torch.tensor([x for y,x in enumerate(input_ids_list) if y in batch_ids]).to(device)
            b_input_mask = torch.tensor([x for y,x in enumerate(att_mask_list) if y in batch_ids]).to(device)
            b_input_images1 = [x for y,x in enumerate(images1_list) if y in batch_ids]
            b_input_images2 = [x for y,x in enumerate(images2_list) if y in batch_ids]
            b_input_coords = [x for y,x in enumerate(coords_list) if y in batch_ids]
            b_labels = torch.tensor([x for y,x in enumerate(labels_list) if y in batch_ids]).to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask, images1=b_input_images1, images2=b_input_images2, labels=b_labels)
            
            loss,logits = outputs

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            points, errors = flat_accuracy(logits, label_ids)
            total_points = total_points + points
            total_errors = total_errors + errors

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clear out the gradients (by default they accumulate)
            model.zero_grad()

            train_acc = total_points/(total_points+total_errors)
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)

            pbar.set_description("accuracy {0:.4f} loss {1:.4f}".format(train_acc, train_loss))

        if save_model:
            torch.save(model, "checkpoints/diagram_checkpoints/dmc_dq_VIT_mcan_L5_VQA_AI2D_"+retrieval_solver+"_e"+str(epoch_i+1)+".pth")
        
        validation_dq_bd(model, raw_data_val, batch_size, device)
        
    print("")
    print("Training complete!")
    
def validation_dq_bd(model, raw_data_val, batch_size, device):

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")
    
    input_ids_list, att_mask_list, images1_list, images2_list, coords_list, labels_list = raw_data_val

    total_points = 0
    total_errors = 0
    val_loss_list = []
    final_res = []

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    sum_aux = 0
    total_aux = 0

    dataset_ids = list(range(len(labels_list)))
    batched_ids = [dataset_ids[k:k+batch_size] for k in range(0, len(dataset_ids), batch_size)]
    pbar = tqdm(batched_ids)
    for batch_ids in pbar:
        # Unpack the inputs from our dataloader
        b_input_ids = torch.tensor([x for y,x in enumerate(input_ids_list) if y in batch_ids]).to(device)
        b_input_mask = torch.tensor([x for y,x in enumerate(att_mask_list) if y in batch_ids]).to(device)
        b_input_images1 = [x for y,x in enumerate(images1_list) if y in batch_ids]
        b_input_images2 = [x for y,x in enumerate(images2_list) if y in batch_ids]
        b_input_coords = [x for y,x in enumerate(coords_list) if y in batch_ids]
        b_labels = torch.tensor([x for y,x in enumerate(labels_list) if y in batch_ids]).to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, images1=b_input_images1, images2=b_input_images2, labels=b_labels)

        loss, logits = outputs

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        for l in logits:
            final_res.append(l)
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        points, errors = flat_accuracy(logits, label_ids)
        total_points = total_points + points
        total_errors = total_errors + errors

        val_loss_list.append(loss.item())

    val_acc = total_points/(total_points+total_errors)
    val_loss = np.mean(val_loss_list)

    print("val_accuracy {0:.4f} val_loss {1:.4f}".format(val_acc, val_loss))
    
    return final_res
    
def generate_interagreement_chart(feats, split):
    models_names = ["IR", "NSPIR", "NNIR"]
    list_elections_max = []
    for fts in feats:
        list_elections = []
        for ft in fts:
            list_elections.append(np.argmax(ft))
        list_elections_max.append(list_elections)
    correlation_matrix = np.zeros((len(list_elections_max),len(list_elections_max)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            i_solver = list_elections_max[i]
            j_solver = list_elections_max[j]
            res = sum(x == y for x, y in zip(i_solver, j_solver))/len(i_solver)
            correlation_matrix[i][j] = res
    print(correlation_matrix)
    f = plt.figure(figsize=(10, 5))
    plt.matshow(correlation_matrix, fignum=f.number, cmap='binary', vmin=0, vmax=1)
    plt.xticks(range(len(models_names)), models_names)
    plt.yticks(range(len(models_names)), models_names)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.savefig(split+'_interagreement.png')
    
def generate_complementarity_chart(feats, labels, split):
    models_names = ["IR", "NSPIR", "NNIR"]
    list_elections_max = []
    for fts in feats:
        list_elections = []
        for ft in fts:
            list_elections.append(np.argmax(ft))
        list_elections_max.append(list_elections)
    correlation_matrix = np.zeros((len(list_elections_max),len(list_elections_max)))
    for i in range(len(feats)):
        for j in range(len(feats)):
            i_solver = list_elections_max[i]
            j_solver = list_elections_max[j]
            points = 0
            totals = 0
            for e1,e2,lab in zip(i_solver,j_solver,labels):
                if e1!=lab:
                    if e2==lab:
                        points = points + 1
                    totals = totals + 1
            res = points/totals
            correlation_matrix[i][j] = res
    print(correlation_matrix)
    f = plt.figure(figsize=(10, 5))
    plt.matshow(correlation_matrix, fignum=f.number, cmap='binary', vmin=0, vmax=1)
    plt.xticks(range(len(models_names)), models_names)
    plt.yticks(range(len(models_names)), models_names)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.savefig(split+'_complementarity.png')

def get_upper_bound(feats, labels):
    points = 0
    for e1,e2,e3,lab in zip(feats[0],feats[1],feats[2],labels):
        if np.argmax(e1)==lab:
            points = points + 1
        else:
            if np.argmax(e2)==lab:
                points = points + 1
            else:
                if np.argmax(e3)==lab:
                    points = points + 1
    upper_bound = points/len(labels)
    return upper_bound

    
def ensembler(feats_train, feats_test, labels_train, labels_test, a, b, c):
    softmax = torch.nn.Softmax(dim=1)
    
    solvers = []
    for feat in feats_train:
        list_of_elems = []
        list_of_labels = []
        for ft, lab in zip(feat, labels_train):
            soft_ft = list(softmax(torch.tensor([ft]))[0].detach().cpu().numpy())
            for i in range(len(soft_ft)):
                list_of_elems.append([ft[i],soft_ft[i]])
                if lab==i:
                    list_of_labels.append(1)
                else:
                    list_of_labels.append(0)
        solvers.append(LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels))
    
    list_of_elems = []
    list_of_labels = []
    for feats1, feats2, feats3, lab in zip(feats_train[0], feats_train[1], feats_train[2], labels_train):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        possible_answers = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output=output1+output2+output3
            list_of_elems.append(output)
            if lab==i:
                list_of_labels.append(1)
            else:
                list_of_labels.append(0)
    final_model = LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels)

    points = 0
    for feats1, feats2, feats3, lab in zip(feats_test[0], feats_test[1], feats_test[2], labels_test):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        possible_answers = []
        outs = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output=output1*a+output2*b+output3*c
            outs.append(output)
        outs = [list(x) for x in outs]
        outs2 = final_model.predict_proba(outs)
        feats = [x[1] for x in outs2]
        outs3 = np.argmax(feats)
        if outs3==lab:
            points = points + 1
    return points/len(labels_test)

def superensembler(feats_train, feats_test, labels_train, labels_test, w1, w2, a, b, c, d, e, f):
    softmax = torch.nn.Softmax(dim=1)
    
    solvers = []
    for feat in feats_train:
        list_of_elems = []
        list_of_labels = []
        for ft, lab in zip(feat, labels_train):
            soft_ft = list(softmax(torch.tensor([ft]))[0].detach().cpu().numpy())
            for i in range(len(soft_ft)):
                list_of_elems.append([ft[i],soft_ft[i]])
                if lab==i:
                    list_of_labels.append(1)
                else:
                    list_of_labels.append(0)
        solvers.append(LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels))
    
    list_of_elems = []
    list_of_labels = []
    for feats1, feats2, feats3, feats4, feats5, feats6, lab in zip(feats_train[0], feats_train[1], feats_train[2], feats_train[3], feats_train[4], feats_train[5], labels_train):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        soft4 = list(softmax(torch.tensor([feats4]))[0].detach().cpu().numpy())
        soft5 = list(softmax(torch.tensor([feats5]))[0].detach().cpu().numpy())
        soft6 = list(softmax(torch.tensor([feats6]))[0].detach().cpu().numpy())
        possible_answers = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output4 = solvers[3].predict_proba([[feats4[i], soft4[i]]])[0]
            output5 = solvers[4].predict_proba([[feats5[i], soft5[i]]])[0]
            output6 = solvers[5].predict_proba([[feats6[i], soft6[i]]])[0]
            output=output1+output2+output3+output4+output5+output6
            list_of_elems.append(output)
            if lab==i:
                list_of_labels.append(1)
            else:
                list_of_labels.append(0)
    final_model = LogisticRegression(solver='liblinear', random_state=42, multi_class='auto').fit(list_of_elems, list_of_labels)
           
    points = 0
    for feats1, feats2, feats3, feats4, feats5, feats6, lab in zip(feats_test[0], feats_test[1], feats_test[2], feats_test[3], feats_test[4], feats_test[5], labels_test):
        soft1 = list(softmax(torch.tensor([feats1]))[0].detach().cpu().numpy())
        soft2 = list(softmax(torch.tensor([feats2]))[0].detach().cpu().numpy())
        soft3 = list(softmax(torch.tensor([feats3]))[0].detach().cpu().numpy())
        soft4 = list(softmax(torch.tensor([feats4]))[0].detach().cpu().numpy())
        soft5 = list(softmax(torch.tensor([feats5]))[0].detach().cpu().numpy())
        soft6 = list(softmax(torch.tensor([feats6]))[0].detach().cpu().numpy())
        possible_answers = []
        outs = []
        for i in range(len(soft1)):
            output1 = solvers[0].predict_proba([[feats1[i], soft1[i]]])[0]
            output2 = solvers[1].predict_proba([[feats2[i], soft2[i]]])[0]
            output3 = solvers[2].predict_proba([[feats3[i], soft3[i]]])[0]
            output4 = solvers[3].predict_proba([[feats4[i], soft4[i]]])[0]
            output5 = solvers[4].predict_proba([[feats5[i], soft5[i]]])[0]
            output6 = solvers[5].predict_proba([[feats6[i], soft6[i]]])[0]
            output=w1*(output1*a+output2*b+output3*c)+w2*(output4*d+output5*e+output6*f)
            outs.append(output)
        outs = [list(x) for x in outs]
        outs2 = final_model.predict_proba(outs)
        feats = [x[1] for x in outs2]
        outs3 = np.argmax(feats)
        if outs3==lab:
            points = points + 1
    return points/len(labels_test)