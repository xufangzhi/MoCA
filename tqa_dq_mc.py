from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import sys
import argparse

from aux_methods import get_data_dq, training_dq, get_data_dq_bd, training_dq_bd
from aux_methods import ResnetRobertaBUTD, VIT,ResnetRoberta, VIT_bd, VIT_bd_v2, VIT_bd_v4, VIT_bd_v5, VIT_mcan, VIT_mcan_L

import os
#os.environ['CUDA_VISIBLE_DEVICES']= '0,3'


def load_model():
    # lm_model = RobertaForMaskedLM.from_pretrained('roberta-base').cuda()
    lm_model = torch.load("/data/linqika/xufangzhi/ISAAQ/checkpoints/pretrain_physics+tqa_spanmask_RACE_e2.pth")
    mc_model = RobertaForMultipleChoice.from_pretrained('roberta-large')

    mc_model.roberta.embeddings.load_state_dict(lm_model.roberta.embeddings.state_dict())
    mc_model.roberta.encoder.load_state_dict(lm_model.roberta.encoder.state_dict())

    return mc_model
    
    
def main(argv):
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'] , help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainings', default="checkpoints/AI2D_e12.pth", help='path to the pretrainings model. Default: checkpoints/AI2D_e12.pth')
    parser.add_argument('-b', '--batchsize', default= 1, type=int, help='size of the batches. Default: 1')
    parser.add_argument('-x', '--maxlen', default= 180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-l', '--lr', default= 1e-6, type=float, help='learning rate. Default: 1e-6')
    parser.add_argument('-e', '--epochs', default= 4, type=int, help='number of epochs. Default: 4')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.pretrainings != "":
        model = VIT_mcan_L()
        #model = ResnetRobertaBUTD()
        #model = load_model()
        model.roberta = torch.load("checkpoints/diagram_checkpoints/AI2D_VIT_e12.pth").roberta
        model.classifier = torch.load("checkpoints/diagram_checkpoints/AI2D_VIT_e12.pth").classifier
        #model = torch.load("/data/linqika/xufangzhi/ISAAQ/checkpoints/diagram_checkpoints/VQA_e2.pth")
        
        print("using VIT_mcan")
    else:
        #model = ResnetRoberta()
        model = VIT()
        print("using vit")
    #for param in model.resnet.parameters():
    #    param.requires_grad = False
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    tokenizer = RobertaTokenizer.from_pretrained("/data/linqika/xufangzhi/ViT/ISAAQ/roberta-large/")
    if args.device=="gpu":
        device = torch.device("cuda:2")
        model.to(device)
    if args.device=="cpu":
        device = torch.device("cpu") 
        model.to(device)
    print(device)
    model.zero_grad()
    
    batch_size = args.batchsize
    max_len = args.maxlen
    lr = args.lr
    epochs = args.epochs
    retrieval_solver = args.retrieval
    save_model = args.save

    #raw_data_train = get_data_dq("train", retrieval_solver, tokenizer, max_len)
    #raw_data_val = get_data_dq("val", retrieval_solver, tokenizer, max_len)
    raw_data_train = get_data_dq_bd("train", retrieval_solver, tokenizer, max_len)
    raw_data_val = get_data_dq_bd("val", retrieval_solver, tokenizer, max_len)
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(raw_data_train[-1]) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    #training_dq(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device, save_model)
    training_dq_bd(model, raw_data_train, raw_data_val, optimizer, scheduler, epochs, batch_size, retrieval_solver, device, save_model)
if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main(sys.argv[1:])