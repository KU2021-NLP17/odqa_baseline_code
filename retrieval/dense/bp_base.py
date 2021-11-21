import tqdm
import time, os, pickle
import os.path as p
from itertools import chain

import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset
from transformers import TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from retrieval.dense import DenseRetrieval

def get_retriever_dataset(args):
    if args.retriever.dense_train_dataset not in [
        "train_dataset",
        "squad_kor_v1",
        "bm25_document_questions",
        "bm25_question_documents",
    ]:
        raise FileNotFoundError(f"{args.retriever.dense_train_dataset}은 DenseRetrieval 데이터셋이 아닙니다.")

    if args.retriever.dense_train_dataset == "squad_kor_v1":
        train_dataset = load_dataset(args.retriever.dense_train_dataset)
    else:
        dataset_path = p.join(args.path.train_data_dir, args.retriever.dense_train_dataset)
        assert p.exists(dataset_path), f"{args.retriever.dense_train_dataset}이 경로에 존재하지 않습니다."
        train_dataset = load_from_disk(dataset_path)

    return train_dataset

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class BPRetrieval(DenseRetrieval):
    def get_embedding(self):
        pass
    
    def get_relevant_doc_bulk(self, queries, topk=1):
        pass 

    def _exec_embedding(self):
        pass

    def _load_dataset(self, eval=False):
        pass

    def _train(self, training_args, train_dataset, p_model, q_model, eval_dataset, existed_p_dir, existed_q_dir, skip_epochs=0):
        pass


