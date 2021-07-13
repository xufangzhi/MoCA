from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer, RobertaForMaskedLM
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import sys
import argparse

from aux_methods import get_data_ndq, process_data_ndq, training_ndq
from transformers import RobertaForMaskedLM, RobertaForMultipleChoice

import os
os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'

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
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN', 'Sim'] , help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    parser.add_argument('-t', '--dataset', default='ndq', choices=['ndq', 'dq'], help='dataset to train the model with. Options: ndq or dq. Default: ndq')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainings', default="checkpoints/pretrainings_e4.pth", help='path to the pretrainings model. If empty, the model will be the RobertForMultipleChoice with roberta-large weights. Default: checkpoints/pretrainings_e4.pth')
    parser.add_argument('-b', '--batchsize', default= 4, type=int, help='size of the batches. Default: 1')
    parser.add_argument('-x', '--maxlen', default= 180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-l', '--lr', default= 1e-5, type=float, help='learning rate. Default: 1e-5')
    parser.add_argument('-e', '--epochs', default= 4, type=int, help='number of epochs. Default: 4')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training', action='store_true')
    args = parser.parse_args()
    print(args)
    
    model = load_model()
    #model = RobertaForMultipleChoice.from_pretrained("roberta-large")
    #model.roberta = torch.load("/data/linqika/xufangzhi/ISAAQ/checkpoints/pretrainingsplus_e1.pth").roberta
    #model = RobertaForMultipleChoice.from_pretrained("/data/linqika/xufangzhi/parallel/test/temp_check/lessons_spanmask/")
    #model = RobertaForMaskedLM.from_pretrained("checkpoints/roberta_4.pth")
    #model = torch.load("checkpoints/roberta_4.pth")
    #model = torch.load("checkpoints/pretrain_RACE_e4.pth")
    #if args.pretrainings == "":
    #    print(1)
    #    model = RobertaForMultipleChoice.from_pretrained("roberta-large")
    #else:
    #    print(2)
    #    model = torch.load(args.pretrainings)
    #    print(3)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    if args.device=="gpu":
        device = torch.device("cuda:2")
        model.to(device)
        print(device)
    if args.device=="cpu":
        device = torch.device("cpu") 
        model.cpu() 
    
    model.zero_grad()
    
    batch_size = args.batchsize
    max_len = args.maxlen
    lr = args.lr
    epochs = args.epochs
    retrieval_solver = args.retrieval
    save_model = args.save
    dataset_name = args.dataset

    raw_data_train = get_data_ndq(dataset_name, "train", retrieval_solver, tokenizer, max_len)
    raw_data_val = get_data_ndq(dataset_name, "val", retrieval_solver, tokenizer, max_len)
    
    train_dataloader = process_data_ndq(raw_data_train, batch_size, "train")
    val_dataloader = process_data_ndq(raw_data_val, batch_size, "val")

    
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    training_ndq(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device, save_model, dataset_name)

if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main(sys.argv[1:])