from transformers import RobertaTokenizer
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse

from aux_methods import get_data_tf, process_data_ndq, validation_tf, get_upper_bound, ensembler

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainingslist', default=["checkpoints/tf_roberta_IR_e2.pth", "checkpoints/tf_roberta_NSP_e3.pth", "checkpoints/tf_roberta_e2.pth"], help='list of paths of the pretrainings model. They must be three. Default: checkpoints/tf_roberta_IR_e1.pth, checkpoints/tf_roberta_NSP_e2.pth, checkpoints/tf_roberta_NN_e1.pth')
    parser.add_argument('-x', '--maxlen', default= 64, type=int, help='max sequence length. Default: 64')
    parser.add_argument('-b', '--batchsize', default= 512, type=int, help='size of the batches. Default: 512')
    args = parser.parse_args()
    print(args)
    
    models = [torch.load(args.pretrainingslist[0]), torch.load(args.pretrainingslist[1]), torch.load(args.pretrainingslist[2])]
    retrieval_solvers = ["IR", "NSP", "NN"]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    max_len = args.maxlen
    batch_size = args.batchsize
    
    feats_train = []
    feats_test = []
    for model, retrieval_solver in zip(models, retrieval_solvers):
        if args.device=="gpu":
            device = torch.device("cuda:3")
            print(device)
            model.to(device)
        if args.device=="cpu":
            device = torch.device("cpu") 
            model.cpu()
        model.eval()
        print("\n")
        print(retrieval_solver)
        print("val")
        raw_data_train = get_data_tf("val", retrieval_solver, tokenizer, max_len)
        train_dataloader = process_data_ndq(raw_data_train, batch_size, "val")
        feats_train.append(validation_tf(model, train_dataloader, device))
        labels_train = raw_data_train[-1]
        
        print("test")
        raw_data_test = get_data_tf("test", retrieval_solver, tokenizer, max_len)
        test_dataloader = process_data_ndq(raw_data_test, batch_size, "test")
        feats_test.append(validation_tf(model, test_dataloader, device))
        labels_test = raw_data_test[-1]

        
    upper_bound_train = get_upper_bound(feats_train, labels_train)
    res = ensembler(feats_train, feats_test, labels_train, labels_test)
    print("\nFINAL RESULTS:")
    print("TEST SET: ")
    print(res)

    res = ensembler(feats_test, feats_train, labels_test, labels_train)
    print("VALIDATION SET: ")
    print(res)

if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main(sys.argv[1:])