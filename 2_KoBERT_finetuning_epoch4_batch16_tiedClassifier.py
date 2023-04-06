'''
    structure : NLI 학습 구조
    train data : STS training dataset
'''

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random
import math
from sentence_transformers_.sentence_transformers import losses
from sentence_transformers_.sentence_transformers import models
from sentence_transformers_.sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers_.sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num_epochs', type=int, default=100, help='give a epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='give a batch size for training')
    parser.add_argument('-s', '--evaluation_step', type=int, default=100,
                        help='step parameter for write dev score and save best model during training')
    parser.add_argument('-r', '--seed', type=int, default=33, help='random seed')
    parser.add_argument('--cuda', type=int, default=2, help='available gpu number(only multi-gpu)')
    parser.add_argument('--tied', type=bool, default=True, help='tied bert model during training')
    parser.add_argument('--model', type=str, default='tiedBERTFinetuning', help='give a model name')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    args = parser.parse_args()

    # random seed 설정
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_batch_size = args.batch_size
    model_name = args.model
    tail = '/2_CustomSTS_tiedClassifier/' + model_name.replace("/", "") + f'_epoch_{args.num_epochs}_batch_{args.batch_size}_tied_{str(args.tied)}_lr_{str(args.lr)}'
    model_save_path = './output' + tail


    # Apply mean pooling to get one fixed sized sentence vector
    model = SentenceTransformer("output/CustomSTS_tied_noAct/tiedBERTModel_epoch_4_batch_16_tied_True_lr_1e-05/epoch_1_step_40_all_400")

    logging.info("Read STS train dataset")

    train_samples = []

    with open('../data/KorNLUDatasets/KorSTS/tune_train.tsv', "rt", encoding="utf-8") as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, score = line.split('\t')
            score = score.strip()
            score = float(score) / 5.0
            train_samples.append(InputExample(texts=[s1, s2], label=score))

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

    # loss model을 save된 모델로 update
    train_loss = losses.CustomSoftmaxLoss(model=model,
                                          sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                          num_labels=1, concatenation_sent_difference=False)
    train_loss.load_state_dict(torch.load("output/CustomSTS_tied_noAct/tiedBERTModel_epoch_4_batch_16_tied_True_lr_1e-05/epoch_1_step_40_all_400/loss_model.pt"))
    #Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []

    with open('../data/KorNLUDatasets/KorSTS/tune_dev.tsv', 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, score = line.split('\t')
            score = score.strip()
            score = float(score) / 5.0
            dev_samples.append(InputExample(texts= [s1,s2], label=score))

    # valid dataloader 선언
    valid_dataset = SentencesDataset(dev_samples,model=model)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=train_batch_size)

    # valid evaluator 선언
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

    num_epochs = args.num_epochs

    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    if args.tied:
        for name, param in train_loss.named_parameters():
            if name in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        for name, param in train_loss.named_parameters():
            param.requires_grad = True


    # Train the model
    writer = SummaryWriter('./runs/' + tail)
    torch.cuda.set_device(f'cuda:{args.cuda}')
    best_model_path = model.fit(train_objectives=[(train_dataloader, train_loss)],
                                evaluator=dev_evaluator,
                                epochs=num_epochs,
                                evaluation_steps=args.evaluation_step,
                                optimizer_params={'lr': args.lr},
                                warmup_steps=warmup_steps,
                                output_path=model_save_path,
                                writer=writer,
                                valid_dataloader=valid_dataloader,
                                save_loss_model=True,
                                optimizer_class=torch.optim.SGD,
                                scheduler='constantlr'
              )


    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    test_samples = []
    with open('../data/KorNLUDatasets/KorSTS/tune_test.tsv', 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, score = line.split('\t')
            score = score.strip()
            score = float(score) / 5.0
            test_samples.append(InputExample(texts=[s1,s2], label=score))



    print("\n\n\n")
    print("======================TEST===================")
    print("\n\n\n")
    model = SentenceTransformer(best_model_path)
    print(f"model save path > {best_model_path}")
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
    test_evaluator(model, output_path=model_save_path)