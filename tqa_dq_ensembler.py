from transformers import RobertaTokenizer
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse

from aux_methods import get_data_ndq, process_data_ndq, get_data_dq, validation_ndq, validation_dq, get_upper_bound, superensembler, ensembler, get_data_dq_bd, validation_dq_bd
import os
os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainingslist', default=["/data/linqika/xufangzhi/ViT/ISAAQ/checkpoints/diagram_checkpoints/dmc_ndq_roberta_IR_e4.pth", "/data/linqika/xufangzhi/ViT/ISAAQ/checkpoints/diagram_checkpoints/dmc_ndq_roberta_NSP_e4.pth", "/data/linqika/xufangzhi/ViT/ISAAQ/checkpoints/diagram_checkpoints/dmc_ndq_roberta_NN_e3.pth", "/data/linqika/xufangzhi/ViT/ISAAQ/checkpoints/diagram_checkpoints/dmc_dq_VIT_mcan_L5_VQA_AI2D_IR_e3.pth", "/data/linqika/xufangzhi/ViT/ISAAQ/checkpoints/diagram_checkpoints/dmc_dq_VIT_mcan_L5_VQA_AI2D_NSP_e3.pth", "/data/linqika/xufangzhi/ViT/ISAAQ/checkpoints/diagram_checkpoints/dmc_dq_VIT_mcan_L5_VQA_AI2D_NN_e1.pth"], help='list of paths of the pretrainings model. They must be three. Default: checkpoints/tmc_dq_roberta_IR_e4.pth, checkpoints/tmc_dq_roberta_NSP_e4.pth, checkpoints/tmc_dq_roberta_NN_e2.pth, checkpoints/dmc_dq_roberta_IR_e3.pth, checkpoints/dmc_dq_roberta_NSP_e4.pth, checkpoints/dmc_dq_roberta_NN_e3.pth')
    parser.add_argument('-x', '--maxlen', default= 180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-b', '--batchsize', default= 1, type=int, help='size of the batches. Default: 512')
    args = parser.parse_args()
    print(args)
    #models = [torch.load(args.pretrainingslist[3]), torch.load(args.pretrainingslist[4]), torch.load(args.pretrainingslist[5])]
    #retrieval_solvers = ["IR", "NSP", "NN"]
    #model_types = ["dmc", "dmc", "dmc"]
    
    
    models = [torch.load(args.pretrainingslist[0]), torch.load(args.pretrainingslist[1]), torch.load(args.pretrainingslist[2]), torch.load(args.pretrainingslist[3]), torch.load(args.pretrainingslist[4]), torch.load(args.pretrainingslist[5])]
    retrieval_solvers = ["IR", "NSP", "NN", "IR", "NSP", "NN"]
    model_types = ["tmc", "tmc", "tmc", "dmc", "dmc", "dmc"]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    max_len = args.maxlen
    batch_size = args.batchsize
    
    feats_train = []
    feats_val = []
    feats_test = []
    for model, model_type, retrieval_solver in zip(models, model_types, retrieval_solvers):
        if args.device=="gpu":
            device = torch.device("cuda:2")
            model.to(device)
        if args.device=="cpu":
            device = torch.device("cpu") 
            model.cpu()
        model.eval()
        print("\n")
        print(retrieval_solver)
        if model_type == "dmc":
            #print("train")
            #raw_data_train = get_data_dq_bd("train", retrieval_solver, tokenizer, max_len)
            #feats_train.append(validation_dq_bd(model, raw_data_train, batch_size, device))
            #labels_train = raw_data_train[-1]
            print("val")
            raw_data_val = get_data_dq_bd("val", retrieval_solver, tokenizer, max_len)
            feats_val.append(validation_dq_bd(model, raw_data_val, batch_size, device))
            labels_val = raw_data_val[-1]
            print("test")
            raw_data_test = get_data_dq_bd("test", retrieval_solver, tokenizer, max_len)
            feats_test.append(validation_dq_bd(model, raw_data_test, batch_size, device))
            labels_test = raw_data_test[-1]
        if model_type == "tmc":
            #print("train")
            #raw_data_train = get_data_ndq("dq", "train", retrieval_solver, tokenizer, max_len)
            #train_dataloader = process_data_ndq(raw_data_train, batch_size, "val")
            #feats_train.append(validation_ndq(model, train_dataloader, device))
            #labels_train = raw_data_train[-1]
            print("val")
            raw_data_val = get_data_ndq("dq", "val", retrieval_solver, tokenizer, max_len)
            val_dataloader = process_data_ndq(raw_data_val, batch_size, "val")
            feats_val.append(validation_ndq(model, val_dataloader, device))
            labels_val = raw_data_val[-1]
            print("test")
            raw_data_test = get_data_ndq("dq", "test", retrieval_solver, tokenizer, max_len)
            test_dataloader = process_data_ndq(raw_data_test, batch_size, "test")
            feats_test.append(validation_ndq(model, test_dataloader, device))
            labels_test = raw_data_test[-1]
   
    #a, b, c = 0.4, 0.5, 0.1
    #d, e, f = 0.5, 0.3, 0.2
    #weight_list = [(0.4, 0.6), (0.5, 0.5), (0.6,0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    #for (w1,w2) in weight_list:
    #    res = superensembler(feats_val, feats_test, labels_val, labels_test, w1, w2, a, b, c, d, e, f)
    #    print("\nFINAL RESULTS:")
    #    print("TEST SET: ")
    #    print(res)
   
    #    res = superensembler(feats_test, feats_val, labels_test, labels_val, w1, w2, a, b, c, d, e, f)
    #    print("VALIDATION SET: ")
    #    print(res)

    weight_list = [(7,3,0),(4,5,1),(5,3,2),(5,5,0),(6,4,0),(5,4,1),(3,5,2),(1,1,1),(4,6,0),(5,2,3),(6,1,3),(6,2,2),(5,1,4)]
    for (a, b, c) in weight_list:
        res = ensembler(feats_test, feats_val, labels_test, labels_val, a, b, c)
        print("Weight:",a,b,c)
        print("VALIDATION SET: ")
        print(res)
        res = ensembler(feats_val, feats_test, labels_val, labels_test, a, b, c)
        print("TEST SET: ")
        print(res)
        
if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main(sys.argv[1:])