from tqdm.auto import tqdm, trange
import pickle
import random
import time
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

class DpRetrieval(DenseRetrieval):
    def __init__(self, args):
        super().__init__(args)  # wiki context load
        self.mappings = []
        self.window_size = 20
        self.sample_per_phrase = 4

        save_dir = p.join(args.path.embed, self.name)
        self.mappings_path = p.join(save_dir, "mappings.bin")

    def get_embedding(self):
        if p.isfile(self.embed_path) and p.isfile(self.encoder_path) and not self.args.retriever.retrain:
            with open(self.embed_path, "rb") as f:
                self.p_embedding = pickle.load(f)

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
        self.encoder.eval()  # question encoder
        self.encoder.cuda()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(
                queries, padding="longest", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            q_embedding = self.encoder(**q_seqs_val)
            q_embedding.squeeze_()  # in-place
            q_embedding = q_embedding.cpu().detach().numpy()

        # p_embedding: numpy, q_embedding: numpy
        result = np.matmul(q_embedding, self.p_embedding.T)
        phrase_indices = np.argsort(result, axis=1)[:, -topk:][:, ::-1]
        doc_indices = [[self.mappings[phrase_indices[i][j]] for j in range(len(phrase_indices[i]))] for i in range(len(phrase_indices))]
        doc_scores = []

        for i in range(len(phrase_indices)):
            doc_scores.append(result[i][[phrase_indices[i].tolist()]])

        return doc_scores, doc_indices

    def _exec_embedding(self):
        p_encoder, q_encoder = self._load_model()

        train_dataset, eval_dataset = self._load_dataset(eval=True)

        args = TrainingArguments(
            output_dir="densephrase_retrieval",
            evaluation_strategy="epoch",
            learning_rate=self.args.retriever.learning_rate,
            per_device_train_batch_size=self.args.retriever.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.retriever.per_device_eval_batch_size,
            num_train_epochs=self.args.retriever.num_train_epochs,
            weight_decay=self.args.retriever.weight_decay,
            gradient_accumulation_steps=self.args.retriever.gradient_accumulation_steps,
        )

        p_encoder, q_encoder = self._train(args, train_dataset, p_encoder, q_encoder, eval_dataset)
        p_embedding = []
        mappings = []

        for idx, passage in enumerate(tqdm(self.contexts)):  # wiki
            splitted = passage.split()
            for i in range(len(splitted) // self.window_size * 2):
                phrase = ' '.join(splitted[i*(self.window_size // 2):(i+2)*(self.window_size //2)])
                phrase = self.tokenizer(
                    phrase, padding="max_length", truncation=True, max_length=self.window_size, return_tensors="pt"
                ).to("cuda")
                p_emb = p_encoder(**phrase).to("cpu").detach().numpy()
                p_embedding.append(p_emb)
                mappings.append(idx)

        p_embedding = np.array(p_embedding).squeeze()  # numpy
        return p_embedding, q_encoder, mappings

class DPTrainMixin:
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
        
        print('Dataset prepare success')
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
    
    def _train(self, training_args, train_dataset, p_model, q_model, eval_dataset):
        print("TRAINING IN DensePhrase TRAIN MIXIN")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=training_args.per_device_train_batch_size, drop_last=True
        )
        if eval_dataset:
            eval_sampler = RandomSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=training_args.per_device_eval_batch_size
            )

        optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)

        t_total = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=t_total
        )

        global_step = 0

        p_model.train()
        q_model.train()

        p_model.zero_grad()
        q_model.zero_grad()

        torch.cuda.empty_cache()

        for epoch in range(training_args.num_train_epochs):
            train_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(train_dataloader):

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                
                embedding_size = batch[0].shape[-1]
                p_inputs = {
                    "input_ids": batch[0].reshape(-1, embedding_size), 
                    "attention_mask": batch[1].reshape(-1, embedding_size), 
                    "token_type_ids": batch[2].reshape(-1, embedding_size)
                }

                q_inputs = {
                    "input_ids": batch[3], 
                    "attention_mask": batch[4], 
                    "token_type_ids": batch[5]
                }
                adder = torch.arange(0, training_args.per_device_train_batch_size).long() * self.sample_per_phrase
                if torch.cuda.is_available():
                    adder = adder.to("cuda")
                label = torch.repeat_interleave(batch[6] + adder, self.sample_per_phrase)

                p_outputs = p_model(**p_inputs)
                q_outputs = q_model(**q_inputs)
                
                q_outputs = torch.repeat_interleave(q_outputs, self.sample_per_phrase, dim=0)

                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))

                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, label)

                loss = loss / training_args.gradient_accumulation_steps
                
#                 print(p_inputs["input_ids"].shape)
#                 print(q_inputs["input_ids"].shape)
#                 print(p_outputs.shape)
#                 print(q_outputs.shape)
                
#                 print(label.shape)
#                 print(label)

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

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss / len(train_dataloader):.4f}")

            p_model.train()
            q_model.train()

        return p_model, q_model