from glob import glob
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

        emb_size = self.p_embedding.shape[1]
        self.p_embedding = np.unpackbits(self.p_embedding).reshape(-1, emb_size * 8).astype(np.float32)
        self.p_embedding = self.p_embedding * 2 - 1

    def get_relevant_doc_bulk(self, queries, topk=1):
        self.encoder.eval()  # question encoder
        self.encoder.cuda()

        with torch.no_grad():
            q_seqs_val = self.tokenizer(
                queries, padding="longest", truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")
            q_embedding = self.encoder(**q_seqs_val)
            q_embedding.squeeze_()  # in-place
            bin_q_emb = self.encoder.convert_to_binary_code(q_embedding).cpu().detach().numpy()
            q_emb = q_embedding.cpu().detach().numpy()

        num_queries = q_emb.shape[0] #
        result = np.matmul(bin_q_emb, self.p_embedding.T)   

        doc_indices, doc_scores = [], []

        if not self.args.retriever.rerank:
            phrase_indices = np.argsort(result, axis=1)[:, -topk * 4:][:, ::-1]

            for row in range(phrase_indices):
                tmp_indices, tmp_scores = [], []
                for col in range(len(phrase_indices[row])):
                    if self.mappings[phrase_indices[row][col]] in tmp_indices: # remove duplicate
                        continue
                    
                    tmp_indices.append(self.mappings[phrase_indices[row][col]])
                    tmp_scores.append(result[row][col])

                    if len(tmp_indices) > topk: # only top_k is needed
                        break

                doc_indices.append(tmp_indices)
                doc_scores.append(tmp_scores)
            
            return doc_scores, doc_indices

        # 1. Generate binary_k candidates by comparing hq with hp
        binary_k = self.args.retriever.binary_k
        cand_indices = np.argsort(result, axis=1)[:, -binary_k:][:, ::-1]

        # 2. Choose top k from the candidates by comparing eq with hp
        cand_p_emb = self.p_embedding[cand_indices] # camd_p_emb.shape = [num_quires, binary_k, embedding_size] 
        scores = np.einsum("ijk,ik->ij", cand_p_emb, q_emb) # [num_quires, binary_k, embedding_size] @ [num_queries, embedding_size, 1] = [num_queries, binary_k, 1]
        sorted_indices = np.argsort(-scores) # [num_queries, topk]

        for row in range(num_queries):
            tmp_indices, tmp_scores = [], []
            for col in sorted_indices.flatten():
                if self.mappings[cand_indices[row][col]] in tmp_indices: # remove duplicate
                    continue
                
                tmp_indices.append(self.mappings[cand_indices[row][col]])
                tmp_scores.append(scores[row][col])

                if len(tmp_indices) > topk: # only top_k is needed
                    break

            doc_indices.append(tmp_indices)
            doc_scores.append(tmp_scores)

        return doc_scores, doc_indices

    def _exec_embedding(self):
        p_encoder, q_encoder = self._load_model()

        # Load existed train dataset
        if p.isfile(self.args.data.preprocessed_trainset_dir):
            with open(self.args.data.preprocessed_trainset_dir, "rb") as f:
                train_dataset = pickle.load(f)
                eval_dataset = None
            
        else:
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

        checkpoint_path = self.args.retriever.checkpoint_path

        p_encoder, q_encoder = self._train(args, train_dataset, p_encoder, q_encoder, eval_dataset, checkpoint_path)
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

        # Save trian dataset 
        try:
            with open(p.join(self.save_dir, "train_dataset.bin"), "wb") as f:
                pickle.dump(train_dataset, f)
            print("Saved train dataset successfully")
        except:
            print("Can't save train dataset")
        
        return train_dataset, eval_dataset

    def _train(self, training_args, train_dataset, p_model, q_model, eval_dataset, checkpoint_path):
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

        optimizer_grouped_parameters = [{"params": p_model.parameters()}, {"params": q_model.parameters()}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)

        t_total = len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
        warmup_steps = int(training_args.warmup_ratio * t_total)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        global_step, start_epoch = 0, 0

        # Load checkpoint
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            p_model.load_state_dict(checkpoint['p_model_state_dict'])
            q_model.load_state_dict(checkpoint['q_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            global_step = checkpoint['global_step']
            start_epoch = checkpoint['epoch'] + 1
            print("Loaded the checkpoint successfully!")

        p_model.train()
        q_model.train()
        p_model.training = True
        q_model.training = True

        p_model.zero_grad()
        q_model.zero_grad()

        torch.cuda.empty_cache()

        # Set timezone
        os.environ['TZ'] = 'Japan'
        time.tzset()
        
        # Make a save folder
        intmd_save_dir = p.join(self.save_dir, f"{time.ctime()}")
        if not p.exists(intmd_save_dir):
            os.mkdir(intmd_save_dir)

        for epoch in range(start_epoch, training_args.num_train_epochs):
            train_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(tqdm(train_dataloader)):
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

            print(f"\nEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss / len(train_dataloader):.4f}")

            # Save passsage and query encoder
            print(f"Save a checkpoint in {intmd_save_dir}...")
            try:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'p_model_state_dict': p_model.state_dict(),
                    'q_model_state_dict': q_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, p.join(intmd_save_dir, f"{self.name}-checkpoint.tar"))
                print("Saved Successfully!")
            except:
                print("Failed to save the checkpoint")

        p_model.eval()
        q_model.eval()
        p_model.training = False
        q_model.training = False

        return p_model, q_model



