import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import wandb
from datetime import datetime
from torch.utils.data import Dataset
import time



class ValTrainDataSet(Dataset):
    def __init__(self,dataset, tokenizer, max_length):
        self.max_length = max_length

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.claim_ids = list(self.dataset.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        processed_text = data["claim_text"].lower()
        return [processed_text, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        queries = []
        datas = []
        evidences = []
        claim_ids = []
        for query, data, claim_id in batch:
            queries.append(query)
            datas.append(data)

            evidences.append(data["evidences"])
            claim_ids.append(claim_id)

        query_text = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["query_input_ids"] = query_text.input_ids
        batch_encoding["query_attention_mask"] = query_text.attention_mask

        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        batch_encoding["evidences"] = evidences
        return batch_encoding
    
class EvidenceDataset(Dataset):
    def __init__(self,dataset, tokenizer, max_length):
        self.max_length = max_length

        self.evidences = dataset


        self.tokenizer = tokenizer
        self.evidences_ids = list(self.evidences.keys())

    def __len__(self):
        return len(self.evidences_ids)

    def __getitem__(self, idx):
        evidences_id = self.evidences_ids[idx]
        evidence = self.evidences[evidences_id]
        return [evidences_id, evidence]

    def collate_fn(self, batch):
        evidences_ids = []
        evidences = []

        for evidences_id, evidence in batch:
            evidences_ids.append(evidences_id)
            evidences.append(evidence.lower())

        evidences_text = self.tokenizer(
            evidences,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["evidence_input_ids"] = evidences_text.input_ids
        batch_encoding["evidence_attention_mask"] = evidences_text.attention_mask
        batch_encoding["evidences_ids"] = evidences_ids
        return batch_encoding

class TrainDataset(Dataset):
    def __init__(
        self, dataset,evidenceDataset, tokenizer, max_length
    ):
        self.max_length = max_length
        self.dataset = dataset
        self.evidences = evidenceDataset
        self.tokenizer = tokenizer
        self.claim_ids = list(self.dataset.keys())
        self.evidence_ids = list(self.evidences.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        processed_query = data["claim_text"].lower()
        evidences = []
        for evidence_id in data["evidences"]:
            evidences.append(evidence_id)
        
        negative_evidences = data["negative_evidences"]
        return [processed_query, evidences, negative_evidences]


    def collate_fn(self, batch):
        queries = []
        evidences = []
        labels = []
        negative_evidences = []
        for query, evidence, negative_evidence in batch:
            queries.append(query)
            evidences.extend(evidence)
            negative_evidences.extend(negative_evidence)
            labels.append(len(evidence))
        evidences.extend(negative_evidences)

        evidences_text = [
            self.evidences[evidence_id].lower() for evidence_id in evidences
        ]
        query_text = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        evidence_text = self.tokenizer(
            evidences_text,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["query_input_ids"] = query_text.input_ids
        batch_encoding["evidence_input_ids"] = evidence_text.input_ids
        batch_encoding["query_attention_mask"] = query_text.attention_mask
        batch_encoding["evidence_attention_mask"] = evidence_text.attention_mask
        batch_encoding["labels"] = labels
        return batch_encoding

class ValDatasetForVal(Dataset):
    def __init__(self,dataset, tokenizer, max_length):
        self.max_length = max_length

      
        self.dataset = dataset
     

        self.tokenizer = tokenizer
        self.claim_ids = list(self.dataset.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        processed_text = data["claim_text"].lower()
        return [processed_text, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        queries = []
        datas = []
        evidences = []
        claim_ids = []
        for query, data, claim_id in batch:
            queries.append(query)
            datas.append(data)

            evidences.append(data["evidences"])
            claim_ids.append(claim_id)

        query_text = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["query_input_ids"] = query_text.input_ids
        batch_encoding["query_attention_mask"] = query_text.attention_mask

        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids

        batch_encoding["evidences"] = evidences
        return batch_encoding

class TestDataset(Dataset):
    def __init__(self,dataset,  tokenizer, max_length):
        self.max_length = max_length

      
        self.dataset = dataset
     

        self.tokenizer = tokenizer
        self.claim_ids = list(self.dataset.keys())
       

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        data = self.dataset[self.claim_ids[idx]]
        processed_text = data["claim_text"].lower()
        return [processed_text, data, self.claim_ids[idx]]

    def collate_fn(self, batch):
        queries = []
        datas = []
        claim_ids = []
        for query, data, claim_id in batch:
            queries.append(query)
            datas.append(data)
        
            claim_ids.append(claim_id)

        query_text = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["query_input_ids"] = query_text.input_ids
        batch_encoding["query_attention_mask"] = query_text.attention_mask

        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids
    
        return batch_encoding

wandb.init(project="nlp assignment 3", name="retrieval")

train = "train-claims.json"
with open (train, "r") as json_file:
    train_dataset = json.load(json_file)
evidence = "evidence.json"
with open (evidence, "r") as json_file:
    evidence_dataset = json.load(json_file)
val = "dev-claims.json"
with open (val, "r") as json_file:
    val_dataset = json.load(json_file)
test = "test-claims-unlabelled.json"
with open (val, "r") as json_file:
    test_dataset = json.load(json_file)

print(type(train_dataset))
claims = []
for key in train_dataset:
    claims.append(train_dataset[key]["claim_text"])
print(len(claims))
longestLengthIndex, longestLength = max(enumerate(claims), key=lambda x: len(x[1]))
print(longestLengthIndex)
print(longestLength)
print(len(longestLength))
shortestLengthIndex, shortestLength = min(enumerate(claims), key=lambda x: len(x[1]))
print(shortestLengthIndex)
print(shortestLength)
print(len(shortestLength))

valclaims = []
for key in val_dataset:
    valclaims.append(val_dataset[key]["claim_text"])
print(len(valclaims))
longestLengthIndex, longestLength = max(enumerate(valclaims), key=lambda x: len(x[1]))
print(longestLengthIndex)
print(longestLength)
print(len(longestLength))
shortestLengthIndex, shortestLength = min(enumerate(valclaims), key=lambda x: len(x[1]))
print(shortestLengthIndex)
print(shortestLength)
print(len(shortestLength))


print(len(evidence_dataset))
longestLengthIndex, longestLength = max(enumerate(evidence_dataset), key=lambda x: len(x[1]))
print(longestLengthIndex)
print(longestLength)
print(len(longestLength))
shortestLengthIndex, shortestLength = min(enumerate(evidence_dataset), key=lambda x: len(x[1]))
print(shortestLengthIndex)
print(shortestLength)
print(len(shortestLength))


testclaims = []
for key in test_dataset:
    testclaims.append(test_dataset[key]["claim_text"])
print(len(testclaims))
longestLengthIndex, longestLength = max(enumerate(testclaims), key=lambda x: len(x[1]))
print(longestLengthIndex)
print(longestLength)
print(len(longestLength))
shortestLengthIndex, shortestLength = min(enumerate(testclaims), key=lambda x: len(x[1]))
print(shortestLengthIndex)
print(shortestLength)
print(len(shortestLength))





torch.manual_seed(600)
torch.cuda.manual_seed_all(600)
np.random.seed(600)
random.seed(600)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_set = ValTrainDataSet(train_dataset,tokenizer, 256)
evidence_set = EvidenceDataset(evidence_dataset,tokenizer, 256)

train_dataloader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=False,
    collate_fn=train_set.collate_fn,
)

evidence_dataloader = DataLoader(
    evidence_set,
    batch_size=128,
    shuffle=False,
    collate_fn=evidence_set.collate_fn,
)

encoder_model = RobertaModel.from_pretrained("roberta-base")
encoder_model.cuda()
encoder_model.eval()

theTotalEvidenceIDs = []
evidenceEmbeddingsData = []
for batch in tqdm(evidence_dataloader):
    for n in batch.keys():
        if n in [
            "query_input_ids",
            "evidence_input_ids",
            "query_attention_mask",
            "evidence_attention_mask",
        ]:
            batch[n] = batch[n].cuda()
    evidence_last = encoder_model(
        input_ids=batch["evidence_input_ids"],
        attention_mask=batch["evidence_attention_mask"],
    ).last_hidden_state
    evidence_embedding_data = evidence_last[:, 0, :].detach()
    evidence_embedding_data_cpu = torch.nn.functional.normalize(
        evidence_embedding_data, p=2, dim=1
    ).cpu()
    del evidence_embedding_data, evidence_last
    evidenceEmbeddingsData.append(evidence_embedding_data_cpu)
    theTotalEvidenceIDs.extend(batch["evidences_ids"])
evidenceEmbeddingsData = torch.cat(evidenceEmbeddingsData, dim=0).t()

result = {}
for batch in tqdm(train_dataloader):
    for n in batch.keys():
        if n in [
            "query_input_ids",
            "evidence_input_ids",
            "query_attention_mask",
            "evidence_attention_mask",
        ]:
            batch[n] = batch[n].cuda()
    query_last = encoder_model(
        input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]
    ).last_hidden_state
    claim_embeddings_data = query_last[:, 0, :]
    claim_embeddings_data = torch.nn.functional.normalize(claim_embeddings_data, p=2, dim=1).cpu()
    scores = torch.mm(claim_embeddings_data, evidenceEmbeddingsData)
    pickTopSimilarities = torch.topk(scores, k=64, dim=1).indices.tolist()
    for idx, data in enumerate(batch["datas"]):
        negative_evidences = []
        for i in pickTopSimilarities[idx]:
            if theTotalEvidenceIDs[i] not in batch["evidences"][idx]:
                negative_evidences.append(theTotalEvidenceIDs[i])
        data["negative_evidences"] = negative_evidences
        result[batch["claim_ids"][idx]] = data
fout = open("positive-and-negative.json", "w")
json.dump(result, fout)
fout.close()

time.sleep(5)

trainPN = "positive-and-negative.json.json"
with open (train, "r") as json_file:
    trainPN_dataset = json.load(json_file)

pos_and_neg_set = TrainDataset(trainPN_dataset,evidence_dataset,tokenizer, 256)
val_set = ValDatasetForVal(val_dataset,tokenizer, 256)
evidence_set = EvidenceDataset(evidence_dataset,tokenizer, 256)


dataloader = DataLoader(
    pos_and_neg_set,
    batch_size=16,
    shuffle=True,
    collate_fn=pos_and_neg_set.collate_fn,
)
val_dataloader = DataLoader(
    val_set, batch_size=16, shuffle=False, collate_fn=val_set.collate_fn
)
evidence_dataloader = DataLoader(
    evidence_set,
    batch_size=128,
    shuffle=False,
    collate_fn=evidence_set.collate_fn,
)

encoder_model_for_train = RobertaModel.from_pretrained("roberta-base")

encoder_model_for_train.cuda()

encoder_optimizer = optim.Adam(encoder_model_for_train.parameters())

for param_group in encoder_optimizer.param_groups:
    param_group["lr"] = 5e-5
encoder_optimizer.zero_grad()
step_cnt = 0
all_step_cnt = 0
avg_loss = 0
maximum_f_score = 0

print("\nEvaluate:\n")
encoder_model_for_train.eval()
theTotalEvidenceIDs = []
evidenceEmbeddingsData = []
for batch in tqdm(evidence_dataloader):
    for n in batch.keys():
        if n in [
            "query_input_ids",
            "evidence_input_ids",
            "query_attention_mask",
            "evidence_attention_mask",
        ]:
            batch[n] = batch[n].cuda()
    evidence_last = encoder_model_for_train(
        input_ids=batch["evidence_input_ids"],
        attention_mask=batch["evidence_attention_mask"],
    ).last_hidden_state
    
    evidence_embedding_data = evidence_last[:, 0, :].detach()
    evidence_embedding_data_cpu = torch.nn.functional.normalize(
        evidence_embedding_data, p=2, dim=1
    ).cpu()
    del evidence_embedding_data, evidence_last
    evidenceEmbeddingsData.append(evidence_embedding_data_cpu)
    theTotalEvidenceIDs.extend(batch["evidences_ids"])
evidenceEmbeddingsData = torch.cat(evidenceEmbeddingsData, dim=0).t()

f_value = []
for batch in tqdm(val_dataloader):
    for n in batch.keys():
        if n in [
            "query_input_ids",
            "evidence_input_ids",
            "query_attention_mask",
            "evidence_attention_mask",
        ]:
            batch[n] = batch[n].cuda()
    query_last = encoder_model_for_train(
        input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]
    ).last_hidden_state
    claim_embeddings_data = query_last[:, 0, :]
    claim_embeddings_data = torch.nn.functional.normalize(claim_embeddings_data, p=2, dim=1).cpu()
    scores = torch.mm(claim_embeddings_data, evidenceEmbeddingsData)
    pickTopSimilarities = torch.topk(scores, k=5, dim=1).indices.tolist()

    for idx, data in enumerate(batch["datas"]):
        evidence_correct = 0
        pred_evidences = [theTotalEvidenceIDs[i] for i in pickTopSimilarities[idx]]
        for evidence_id in batch["evidences"][idx]:
            if evidence_id in pred_evidences:
                evidence_correct += 1
        if evidence_correct > 0:
            evidence_recall = float(evidence_correct) / len(batch["evidences"][idx])
            evidence_precision = float(evidence_correct) / len(pred_evidences)
            evidence_fscore = (2 * evidence_precision * evidence_recall) / (
                evidence_precision + evidence_recall
            )
        else:
            evidence_fscore = 0
        f_value.append(evidence_fscore)

fscore = np.mean(f_value)

encoder_model_for_train.train()
wandb.log({"fscore": fscore}, step=all_step_cnt)
for epoch in range(15):
    epoch_step = 0
    for i, batch in enumerate(tqdm(dataloader)):
        for n in batch.keys():
            if n in [
                "query_input_ids",
                "evidence_input_ids",
                "query_attention_mask",
                "evidence_attention_mask",
            ]:
                batch[n] = batch[n].cuda()
        step_cnt += 1
        claim_embeddings_datas = encoder_model_for_train(
            input_ids=batch["query_input_ids"],
            attention_mask=batch["query_attention_mask"],
        ).last_hidden_state
        evidenceEmbeddingsData = encoder_model_for_train(
            input_ids=batch["evidence_input_ids"],
            attention_mask=batch["evidence_attention_mask"],
        ).last_hidden_state
        claim_embeddings_datas = claim_embeddings_datas[:, 0, :]
        evidenceEmbeddingsData = evidenceEmbeddingsData[:, 0, :]
        claim_embeddings_datas = torch.nn.functional.normalize(claim_embeddings_datas, p=2, dim=1)
        evidenceEmbeddingsData = torch.nn.functional.normalize(
            evidenceEmbeddingsData, p=2, dim=1
        )

        cos_sims = torch.mm(claim_embeddings_datas, evidenceEmbeddingsData.t())
        scores = -torch.nn.functional.log_softmax(cos_sims / 0.05, dim=1)

        loss = []
        start_idx = 0
        for idx, label in enumerate(batch["labels"]):
            end_idx = start_idx + label
            cur_loss = torch.mean(scores[idx, start_idx:end_idx])
            loss.append(cur_loss)
            start_idx = end_idx

        loss = torch.stack(loss).mean()
        loss = loss / 2
        loss.backward()

        avg_loss = avg_loss + loss.item()
        if step_cnt == 2:
            nn.utils.clip_grad_norm_(encoder_model_for_train.parameters(), 1)
            step_cnt = 0
            epoch_step += 1
            all_step_cnt += 1
            if all_step_cnt <= 150:
                lr = all_step_cnt * (5e-5 - 2e-8) / 150 + 2e-8
            else:
                lr = 5e-5 - (all_step_cnt - 150) * 1e-8
            for param_group in encoder_optimizer.param_groups:
                param_group["lr"] = lr
            encoder_optimizer.step()
            encoder_optimizer.zero_grad()

        if all_step_cnt % 10 == 0 and step_cnt == 0:
            if all_step_cnt <= 150:
                lr = all_step_cnt * (5e-5 - 5e-8) / 150 + 5e-8
            else:
                lr = 5e-5 - (all_step_cnt - 150) * 1e-8

            wandb.log({"learning_rate": lr}, step=all_step_cnt)
            wandb.log({"loss": avg_loss / 10}, step=all_step_cnt)

            avg_loss = 0
        del loss, cos_sims, claim_embeddings_datas, evidenceEmbeddingsData

        if all_step_cnt % 50 == 0 and all_step_cnt != 0 and step_cnt == 0:
            encoder_model_for_train.eval()
            theTotalEvidenceIDs = []
            evidenceEmbeddingsData = []
            for batch in tqdm(evidence_dataloader):
                for n in batch.keys():
                    if n in [
                        "query_input_ids",
                        "evidence_input_ids",
                        "query_attention_mask",
                        "evidence_attention_mask",
                    ]:
                        batch[n] = batch[n].cuda()
                evidence_last = encoder_model_for_train(
                    input_ids=batch["evidence_input_ids"],
                    attention_mask=batch["evidence_attention_mask"],
                ).last_hidden_state
                evidence_embedding_data = evidence_last[:, 0, :].detach()
                evidence_embedding_data_cpu = torch.nn.functional.normalize(
                    evidence_embedding_data, p=2, dim=1
                ).cpu()
                del evidence_embedding_data, evidence_last
                evidenceEmbeddingsData.append(evidence_embedding_data_cpu)
                theTotalEvidenceIDs.extend(batch["evidences_ids"])
            evidenceEmbeddingsData = torch.cat(evidenceEmbeddingsData, dim=0).t()

            f_value = []
            for batch in tqdm(val_dataloader):
                for n in batch.keys():
                    if n in [
                        "query_input_ids",
                        "evidence_input_ids",
                        "query_attention_mask",
                        "evidence_attention_mask",
                    ]:
                        batch[n] = batch[n].cuda()
                query_last = encoder_model_for_train(
                    input_ids=batch["query_input_ids"],
                    attention_mask=batch["query_attention_mask"],
                ).last_hidden_state
                claim_embeddings_data = query_last[:, 0, :]
                claim_embeddings_data = torch.nn.functional.normalize(
                    claim_embeddings_data, p=2, dim=1
                ).cpu()
                scores = torch.mm(claim_embeddings_data, evidenceEmbeddingsData)
                pickTopSimilarities = torch.topk(scores, k=5, dim=1).indices.tolist()
                for idx, data in enumerate(batch["datas"]):
                    evidence_correct = 0
                    pred_evidences = [theTotalEvidenceIDs[i] for i in pickTopSimilarities[idx]]
                    for evidence_id in batch["evidences"][idx]:
                        if evidence_id in pred_evidences:
                            evidence_correct += 1
                    if evidence_correct > 0:
                        evidence_recall = float(evidence_correct) / len(
                            batch["evidences"][idx]
                        )
                        evidence_precision = float(evidence_correct) / len(
                            pred_evidences
                        )
                        evidence_fscore = (2 * evidence_precision * evidence_recall) / (
                            evidence_precision + evidence_recall
                        )
                    else:
                        evidence_fscore = 0
                    f_value.append(evidence_fscore)
            fscore = np.mean(f_value)

            encoder_model_for_train.train()
            wandb.log({"f_score": fscore}, step=all_step_cnt)
            if fscore > maximum_f_score:
                maximum_f_score = fscore




test_set = TestDataset(test_dataset,tokenizer, 256)
testdataloader = DataLoader(
    test_set,
    batch_size=16,
    shuffle=False,
    collate_fn=test_set.collate_fn,
)

theTotalEvidenceIDs = []
evidenceEmbeddingsData = []
for batch in tqdm(evidence_dataloader):
    for n in batch.keys():
        if n in [
            "query_input_ids",
            "evidence_input_ids",
            "query_attention_mask",
            "evidence_attention_mask",
        ]:
            batch[n] = batch[n].cuda()
    evidence_last = encoder_model_for_train(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
    evidence_embedding_data = evidence_last[:, 0, :].detach()
    evidence_embedding_data_cpu = torch.nn.functional.normalize(evidence_embedding_data, p=2, dim=1).cpu()
    del evidence_embedding_data, evidence_last
    evidenceEmbeddingsData.append(evidence_embedding_data_cpu)
    theTotalEvidenceIDs.extend(batch["evidences_ids"])
evidenceEmbeddingsData = torch.cat(evidenceEmbeddingsData, dim=0).t()

result = {}
for batch in tqdm(testdataloader):
    for n in batch.keys():
        if n in [
            "query_input_ids",
            "evidence_input_ids",
            "query_attention_mask",
            "evidence_attention_mask",
        ]:
            batch[n] = batch[n].cuda()
    query_last = encoder_model_for_train(
        input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]
    ).last_hidden_state
    claim_embeddings_data = query_last[:, 0, :]
    claim_embeddings_data = torch.nn.functional.normalize(claim_embeddings_data, p=2, dim=1).cpu()
    scores = torch.mm(claim_embeddings_data, evidenceEmbeddingsData)
    pickTopSimilarities = torch.topk(scores, k=5, dim=1).indices.tolist()
    for idx, data in enumerate(batch["datas"]):
        data["evidences"] = [theTotalEvidenceIDs[i] for i in pickTopSimilarities[idx]]
        result[batch["claim_ids"][idx]] = data
fout = open("retrieval-predictions", "w")
json.dump(result, fout)
fout.close()