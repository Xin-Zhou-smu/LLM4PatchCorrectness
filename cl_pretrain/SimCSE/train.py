import sys
import argparse

from tqdm import tqdm
from loguru import logger
import os
import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataloader import TrainDataset, TestDataset, load_sts_data, load_sts_data_unsup, load_diff_data_unsup
from model import SimcseModel, simcse_unsup_loss
from transformers import BertModel, BertConfig, BertTokenizer, AutoTokenizer, AutoModel, AutoConfig


def train(model, train_dl, dev_dl, optimizer, device, save_path, tokenizer, config, epoch, accumulation_steps):
    """模型训练函数"""
    model.train()
    best = -1
    previous_batch_idx = 0
    for ep in range(epoch):
        print('Epoch: ', ep)
        for batch_idx, source in enumerate(tqdm(train_dl), start=1):
            batch_idx += previous_batch_idx
            # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(device)
            attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(device)
            # token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(device)

            # out = model(input_ids, attention_mask, token_type_ids)
            out = model(input_ids, attention_mask)

            loss = simcse_unsup_loss(out, device)
            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            # Optimize every accumulation_steps iterations
            if ( (batch_idx+1) % accumulation_steps == 0) or (batch_idx == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 100 == 0:
                logger.info(f'loss: {loss.item():.4f}')

                # pt_save_directory = save_path + str(batch_idx)
                pt_save_directory = save_path + '/best/'
                if not os.path.exists(pt_save_directory):
                    os.makedirs(pt_save_directory)

                corrcoef = evaluation(model, dev_dl, device)
                logger.info(f"corrcoef: {corrcoef:.4f} in batch: {batch_idx}")
                model.train()
                if best < corrcoef:
                    best = corrcoef
                    logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                    tokenizer.save_pretrained(pt_save_directory)
                    config.save_pretrained(pt_save_directory)
                    torch.save(model.state_dict(), pt_save_directory + '/pytorch_model.bin')

                    ##rename the key in model parameters
                    import copy
                    state_dict = copy.deepcopy(model.state_dict())
                    state_dict2 = copy.deepcopy(model.state_dict())
                    keys = []
                    for key in state_dict:
                        keys.append(key)
                    for k in keys:
                        state_dict[k.split('.', 1)[-1]] = state_dict2[k]
                        state_dict.pop(k)
                    torch.save(state_dict, pt_save_directory + '/pytorch_model.bin')
        previous_batch_idx = batch_idx
        print('updated previous_batch_idx into', previous_batch_idx)



def evaluation(model, dataloader, device):
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            # source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            # source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            source_pred = model(source_input_ids, source_attention_mask)

            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            # target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            # target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            target_pred = model(target_input_ids, target_attention_mask)

            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    print(label_array[0:20])
    # print(sim_tensor.cpu().numpy())
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def main(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train_path_sp = args.data_path + "cnsd-sts-train.txt"
    train_path_unsp = "../sstub_data/code_change_list.pkl"

    # dev_path_sp = args.data_path + "cnsd-sts-dev.txt"
    # test_path_sp = args.data_path + "cnsd-sts-test.txt"
    test_path_sp = args.data_path + "valid.txt"
    # pretrain_model_path = "/data/Learn_Project/Backup_Data/macbert_chinese_pretrained"

    test_data_source = load_sts_data(test_path_sp)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    config = AutoConfig.from_pretrained(args.pretrain_model_path)

    if args.un_supervise:
        ## 无监督的 Unsupervised Training Data
        # train_data_source = load_sts_data_unsup(train_path_unsp)
        train_data_source = load_diff_data_unsup(train_path_unsp)
        train_sents = [data[0] for data in train_data_source]
        train_dataset = TrainDataset(train_sents, tokenizer, max_len=args.max_length)

    # else:
    #     ## 有监督的 Supervised Training Data
    #     train_data_source = load_sts_data(train_path_sp)
    #     # train_sents = [data[0] for data in train_data_source] + [data[1] for data in train_data_source]
    #     train_dataset = TestDataset(train_data_source, tokenizer, max_len=args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    test_dataset = TestDataset(test_data_source, tokenizer, max_len=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"]
    model = SimcseModel(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(
        args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epoch=args.epoch_num
    accumulation_steps = args.accumulation_steps

    save_path = args.save_path + '_' + args.pooler + '_dropout_' + str(args.dropout) + "/"
    print(save_path)

    train(model, train_dataloader, test_dataloader, optimizer, args.device, save_path, tokenizer, config, epoch, accumulation_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', help="gpu or cpu")
    parser.add_argument("--save_path", type=str, default='./model_save_validated')
    parser.add_argument("--un_supervise", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=5e-5) #3e-5
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=float, default=16) #8)
    parser.add_argument("--accumulation_steps", type=float, default=4)  # 8)
    parser.add_argument("--epoch_num", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=480, help="max length of input sentences") #64
    parser.add_argument("--data_path", type=str, default="../bigclonebench_sampled_as_validation_data/")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="microsoft/codebert-base-mlm")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='last-avg', help='which pooler to use')

    args = parser.parse_args()
    logger.add("../log/train.log")
    logger.info(args)
    main(args)
