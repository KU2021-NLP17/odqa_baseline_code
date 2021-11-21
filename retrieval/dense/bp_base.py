import tqdm
from tqdm.auto import tqdm, trange
import random
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
        p_encoder, q_encoder = self._load_model()

        train_dataset, eval_dataset = self._load_dataset(eval=True)

        args = TrainingArguments(
            output_dir="binaryphrase_retrieval",
            evaluation_strategy="epoch",
            learning_rate=self.args.retriever.learning_rate,
            per_device_train_batch_size=self.args.retriever.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.retriever.per_device_eval_batch_size,
            num_train_epochs=self.args.retriever.num_train_epochs,
            weight_decay=self.args.retriever.weight_decay,
            gradient_accumulation_steps=self.args.retriever.gradient_accumulation_steps,
            warmup_ratio=self.args.retriever.warmup_ratio,
        )

        existed_p_dir = self.args.retriever.existed_p_dir
        existed_q_dir = self.args.retriever.existed_q_dir
        skip_epochs = self.args.retriever.skip_epochs

        p_encoder, q_encoder = self._train(
            args, train_dataset, p_encoder, q_encoder, eval_dataset,  existed_p_dir, existed_q_dir, skip_epochs
            )
        p_embedding = []
        mappings = []

        for idx, passage in enumerate(tqdm(self.contexts)):  # wiki
            splitted = passage.split()
            for i in range(len(splitted) // self.window_size * 2):
                phrase = ' '.join(splitted[i*(self.window_size // 2):(i+2)*(self.window_size //2)])
                phrase = self.tokenizer(
                    phrase, padding="max_length", truncation=True, max_length=self.window_size, return_tensors="pt"
                ).to("cuda")
                p_emb = p_encoder(**phrase)
                p_emb = p_encoder.convert_to_binary_code(p_emb).to("cpu").detach().numpy()
                p_emb = np.where(p_emb == -1, 0, p_emb).astype(np.bool)
                p_emb = np.packbits(p_emb).reshape(p_emb.shape[0], -1)
                p_embedding.append(p_emb)
                mappings.append(idx)

        p_embedding = np.array(p_embedding).squeeze()  # numpy
        return p_embedding, q_encoder, mappings


    def _load_dataset(self, eval=False):
        datasets = get_retriever_dataset(self.args)

        train_dataset = datasets["train"]

        # TODO delete
        # train_dataset = train_dataset.select(range(100))

        # with negative examples
        questions = []
        phrases = []
        labels = []
        
        for idx, question in enumerate(tqdm(train_dataset["question"])):
            answer_passage = train_dataset["context"][idx]
            splitted = answer_passage.split()

            for phrase_idx in range(len(splitted) // self.window_size * 2):
                phrase = ' '.join(splitted[phrase_idx*(self.window_size // 2):(phrase_idx+2)*(self.window_size //2)])

                while True:
                    incorrect_passage = random.choice(train_dataset["context"])
                    incorrect_passage_splitted = incorrect_passage.split()
                    if len(incorrect_passage_splitted) // self.window_size >= self.sample_per_phrase:
                        break

                incorrect_phrase_indices = random.sample(range(0, len(incorrect_passage_splitted) // self.window_size), self.sample_per_phrase - 1)
                incorrect_phrases = [' '.join(incorrect_passage_splitted[i*(self.window_size // 2):(i+2)*(self.window_size //2)]) for i in incorrect_phrase_indices]
                
                incorrect_phrases.insert(phrase_idx % self.sample_per_phrase, phrase)

                questions.append(question)
                phrases.append(incorrect_phrases)
                labels.append(phrase_idx % self.sample_per_phrase)
        
        print('The dataset is loaded successfully.')
        print(f'Length : {len(questions)}')

        q_seqs = self.tokenizer(
            questions, padding="longest", truncation=True, max_length=512, return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            list(chain(*phrases)), padding="max_length", truncation=True, max_length=self.window_size, return_tensors="pt"
        )

        embedding_size = p_seqs["input_ids"].shape[-1]
        for k in p_seqs.keys():
            p_seqs[k] = p_seqs[k].reshape(-1, self.sample_per_phrase, embedding_size)
        
        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
            torch.tensor(labels)
        )

        eval_dataset = None
        
        return train_dataset, eval_dataset

    def _train(self, training_args, train_dataset, p_model, q_model, eval_dataset, existed_p_dir, existed_q_dir, skip_epochs=0):
        pass


