from tqdm.auto import tqdm, trange
import time, os, pickle, random
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
    def __init__(self, args):
        super().__init__(args)  # wiki context load
        self.mappings = []
        self.window_size = 20
        self.sample_per_phrase = 4

        self.mappings_path = p.join(self.save_dir, "mappings.bin")

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path) and not self.args.retriever.retrain:
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            emb_size = self.p_embedding.shape[1]
            self.p_embedding = np.unpackbits(self.p_embedding).reshape(-1, emb_size * 8).astype(np.float32)
            self.p_embedding= self.p_embedding * 2 - 1

            with open(self.mappings_path, "rb") as f:
                self.mappings = pickle.load(f)

            self.encoder = self._get_encoder()
            self.encoder.load_state_dict(torch.load(self.encoder_path))
        else:
            self.p_embedding, self.encoder, self.mappings = self._exec_embedding()

            with open(self.embed_path, "wb") as f:
                pickle.dump(self.p_embedding, f)
            
            with open(self.mappings_path, "wb") as f:
                pickle.dump(self.mappings, f)

            torch.save(self.encoder.state_dict(), self.encoder_path)

    def get_relevant_doc_bulk(self, queries, topk=1):
        pass 

    def _exec_embedding(self):
        pass

    def _load_dataset(self, eval=False):
        pass
    
    def _train(self, training_args, train_dataset, p_model, q_model, eval_dataset, existed_p_dir, existed_q_dir, skip_epochs=0):
        print("TRAINING Binary Phrases Retriever")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=training_args.per_device_train_batch_size, drop_last=True
        )
        if eval_dataset:
            eval_sampler = RandomSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=training_args.per_device_eval_batch_size
            )

        if existed_p_dir:
            p_model.load_state_dict(torch.load(existed_p_dir))
        
        if existed_q_dir:
            q_model.load_state_dict(torch.load(existed_p_dir))

        optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)

        t_total = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
        warmup_steps = int(training_args.warmup_ratio * t_total)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        p_model.train()
        q_model.train()
        p_model.training = True
        q_model.training = True

        p_model.zero_grad()
        q_model.zero_grad()

        torch.cuda.empty_cache()

        intmd_save_dir = p.join(self.save_dir, f"{time.ctime()}")
        if not p.exists(intmd_save_dir):
            os.mkdir(intmd_save_dir)

        for epoch in range(training_args.num_train_epochs):
            train_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(train_dataloader):
                # Skip epochs to continue learning
                if epoch < skip_epochs:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    continue

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                embedding_size = batch[0].shape[-1]
                p_inputs = {
                    "input_ids": batch[0].reshape(-1, embedding_size), 
                    "attention_mask": batch[1].reshape(-1, embedding_size),
                    "token_type_ids": batch[2].reshape(-1, embedding_size) 
                    } 
                q_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
            

                p_outputs = p_model(**p_inputs) # shape = [batch_size, embedding_size]
                q_outputs = q_model(**q_inputs) # shape = [batch_size, embedding_size]

                # Convert embeddings to binary code
                p_outputs = p_model.convert_to_binary_code(p_outputs, global_step)
                
                # Match the dimension of queries to the phrases
                q_outputs = torch.repeat_interleave(q_outputs, self.sample_per_phrase, dim=0)
                
                adder = torch.arange(0, training_args.per_device_train_batch_size).long() * self.sample_per_phrase
                if torch.cuda.is_available():
                    adder = adder.to("cuda")
                
                labels = torch.repeat_interleave(batch[6] + adder, self.sample_per_phrase)
                
                scores = torch.matmul(q_outputs, p_outputs.transpose(0, 1))
                dense_loss = F.cross_entropy(scores, labels) # Rerank loss

                binary_q_outputs = q_model.convert_to_binary_code(q_outputs, global_step)
                binary_q_scores = torch.matmul(binary_q_outputs, p_outputs.transpose(0, 1))
                
                if self.args.retriever.use_binary_cross_entropy_loss:
                    binary_loss = F.cross_entropy(binary_q_scores, labels)
                else:
                    pos_mask = binary_q_scores.new_zeros(binary_q_scores.size(), dtype=torch.bool)
                    for n, label in enumerate(labels):
                        pos_mask[n, label] = True
                    pos_bin_scores = torch.masked_select(binary_q_scores, pos_mask)
                    pos_bin_scores = pos_bin_scores.repeat_interleave(p_outputs.size(0) - 1)
                    neg_bin_scores = torch.masked_select(binary_q_scores, torch.logical_not(pos_mask))
                    bin_labels = pos_bin_scores.new_ones(pos_bin_scores.size(), dtype=torch.int64)
                    binary_loss = F.margin_ranking_loss(  
                        pos_bin_scores, neg_bin_scores, bin_labels, self.args.retriever.binary_ranking_loss_margin,
                    )

                loss = binary_loss + dense_loss

                loss = loss / training_args.gradient_accumulation_steps

                print(f"epoch: {epoch + 1:02} step: {step:02} loss: {loss}", end="\r")
                train_loss += loss.item()

                loss.backward()

                if ((step + 1) % training_args.gradient_accumulation_steps) == 0:
                    optimizer.step()
                    scheduler.step()

                    p_model.zero_grad()
                    q_model.zero_grad()

                global_step += 1
                torch.cuda.empty_cache()

            # Skip epochs to continue learning
            if epoch < skip_epochs:
                print(f"skipping {epoch} epoch...")
                continue

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss / len(train_dataloader):.4f}")

            if epoch % 5 == 0:
                print("Save model...")
                torch.save(p_model.state_dict(), p.join(intmd_save_dir, f"{self.name}-p.pth"))
                torch.save(q_model.state_dict(), p.join(intmd_save_dir, f"{self.name}-q.pth"))
                print("Save Success!")

        p_model.eval()
        q_model.eval()
        p_model.training = False
        q_model.training = False

        return p_model, q_model



