import os
import sys
import dill
import logging
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random

import torch
import torch.nn.functional as F
from torch.optim import AdamW as Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

from models.ours import RAREMed
from utils.util import multi_label_metric, ddi_rate_score, get_n_params, create_log_id, logging_config, get_grouped_metrics, \
    get_model_path, get_pretrained_model_path


# evaluate
@torch.no_grad()
def evaluator(args, model, data_val, voc_size, epoch, 
              ddi_adj_path, device, rec_results_path = ''):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    smm_record = []                 # 统计所有推荐结果，最后一起算一个DDI rate
    med_cnt, visit_cnt = 0, 0       # 统计平均药物数量
    recommended_drugs = set()
    loss_val_bce = loss_val_milti = loss_val_ddi = loss_val_sum = 0                # 统计验证集总loss
    len_val = len(data_val)

    rec_results = []
    all_pred_prob = []
    ja_visit = [[] for _ in range(5)]

    for patient in tqdm(data_val, ncols=60, total=len(data_val), desc=f"Evaluation"):
        '''
            y_gt:         v*drug_num, binary
            y_pred_prob:  v*drug_num, predict probability, float(0-1)
            y_pred:       v*drug_num, predict label, binary
            y_pred_label: v*pred_num, predicted drug index, int
        '''
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        visit_weights_patient = []
        
        all_diseases, all_procedures, all_medications = [], [], []   # if test

        for adm in patient: # every admission
            if args.test:
                all_diseases.append(adm[0])
                all_procedures.append(adm[1])
                all_medications.append(adm[2])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp) # correct medicine
            visit_weights.append(adm[3])
            visit_weights_patient.append(adm[3])
        results, loss_ddi = model(patient)
        loss_bce, loss_multi = loss_func(voc_size, patient, results, device)
        loss_val_bce += loss_bce.item()/len_val
        loss_val_milti += loss_multi.item()/len_val
        loss_val_ddi += loss_ddi.item()/len_val
        y_pred_prob = F.sigmoid(results).detach().cpu().numpy()
        for target_output in y_pred_prob:       # each visit
            # prediction med set
            y_pred_tmp = target_output.copy()
            all_pred_prob.append(list(y_pred_tmp))
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            recommended_drugs = set(y_pred_label_tmp) | recommended_drugs #???
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label) # 所有patient的y prediction
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        if args.test:
            if len(patient) < 5:
                ja_visit[len(patient)-1].append(adm_ja)
            else:
                ja_visit[4].append(adm_ja)
            records = [all_diseases, all_procedures, all_medications, y_pred_label, visit_weights_patient, [adm_ja]]
            rec_results.append(records)
                
            
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    if args.test:
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        plot_path = rec_results_path + '/' + 'pred_prob.jpg'
        # plot_hist(all_pred_prob, plot_path)
        ja_result_file = rec_results_path + '/' + 'ja_result.pkl'
        dill.dump(ja_visit, open(ja_result_file, 'wb'))
        for i in range(5):
            logging.info(str(i+1) + f'visit\t mean: {np.mean(ja_visit[i]):.4}, std: {np.std(ja_visit[i]):.4}')
   
    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path) # if (epoch != 0) | (mode=='test')  else 0.0

    get_grouped_metrics(ja, visit_weights)
    logging.info(f'''Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, AVG_MED: {med_cnt / visit_cnt:.4}''')  

    loss_val_sum = (1 - args.weight_multi) * loss_val_bce + args.weight_multi * loss_val_milti + args.weight_ddi * loss_val_ddi

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), med_cnt / visit_cnt,\
          loss_val_bce, loss_val_milti, loss_val_ddi, loss_val_sum

@torch.no_grad()
def evaluator_mask(model, data_val, voc_size, epoch, device, mode='pretrain'):
    model.eval()
    loss_val = 0
    dis_ja_list, dis_prauc_list, dis_p_list, dis_r_list, dis_f1_list = [[] for _ in range(5)]
    pro_ja_list, pro_prauc_list, pro_p_list, pro_r_list, pro_f1_list = [[] for _ in range(5)]
    dis_cnt, pro_cnt, visit_cnt = 0, 0, 0     # 统计平均药物数量
    recommended_drugs = set()
    len_val = len(data_val)
    for batch in tqdm(data_val, ncols=60, desc=mode, total=len_val):
        batch_size = len(batch)
        dis_pred, dis_pred_label = [[] for i in range(2)]
        pro_pred, pro_pred_label = [[] for i in range(2)]
        
        result = model(batch, mode)
        dis_gt = np.zeros((batch_size, voc_size[0]))
        pro_gt = np.zeros((batch_size, voc_size[1]))
        for i in range(batch_size):
            dis_gt[i, batch[i][0]] = 1
            pro_gt[i, batch[i][1]] = 1
        target = np.concatenate((dis_gt, pro_gt), axis=1)
        loss = F.binary_cross_entropy_with_logits(result, torch.tensor(target, device=device))
        loss_val += loss.item()

        dis_logit = result[:, :voc_size[0]]
        pro_logit = result[:, voc_size[0]:]
        dis_pred_prob = F.sigmoid(dis_logit).cpu().numpy()
        pro_pred_prob = F.sigmoid(pro_logit).cpu().numpy()

        visit_cnt += batch_size
        for i in range(batch_size):
            dis_pred_temp = dis_pred_prob[i].copy()
            dis_pred_temp[dis_pred_temp>=0.5] = 1
            dis_pred_temp[dis_pred_temp<0.5] = 0
            dis_pred.append(dis_pred_temp)
            
            dis_pred_label_temp = np.where(dis_pred_temp == 1)[0]
            dis_pred_label.append(sorted(dis_pred_label_temp))
            dis_cnt += len(dis_pred_label_temp)

            pro_pred_temp = pro_pred_prob[i].copy()
            pro_pred_temp[pro_pred_temp>=0.5] = 1
            pro_pred_temp[pro_pred_temp<0.5] = 0
            pro_pred.append(pro_pred_temp)
            
            pro_pred_label_temp = np.where(pro_pred_temp == 1)[0]
            pro_pred_label.append(sorted(pro_pred_label_temp))
            pro_cnt += len(pro_pred_label_temp)

        
        dis_ja, dis_prauc, dis_avg_p, dis_avg_r, dis_avg_f1 = multi_label_metric(
            np.array(dis_gt), np.array(dis_pred), np.array(dis_pred_prob))
        pro_ja, pro_prauc, pro_avg_p, pro_avg_r, pro_avg_f1 = multi_label_metric(
            np.array(pro_gt), np.array(pro_pred), np.array(pro_pred_prob))
        
        dis_ja_list.append(dis_ja)
        dis_prauc_list.append(dis_prauc)
        dis_p_list.append(dis_avg_p)
        dis_r_list.append(dis_avg_r)
        dis_f1_list.append(dis_avg_f1)

        pro_ja_list.append(pro_ja)
        pro_prauc_list.append(pro_prauc)
        pro_p_list.append(pro_avg_p)
        pro_r_list.append(pro_avg_r)
        pro_f1_list.append(pro_avg_f1)

        avg_dis_ja, avg_dis_prauc, avg_dis_p, avg_dis_r, avg_dis_f1 = np.mean(dis_ja_list), np.mean(dis_prauc_list), np.mean(dis_p_list), np.mean(dis_r_list), np.mean(dis_f1_list)
        avg_pro_ja, avg_pro_prauc, avg_pro_p, avg_pro_r, avg_pro_f1 = np.mean(pro_ja_list), np.mean(pro_prauc_list), np.mean(pro_p_list), np.mean(pro_r_list), np.mean(pro_f1_list)
        avg_ja, avg_prauc, avg_p, avg_r, avg_f1 = (avg_dis_ja+avg_pro_ja)/2, (avg_dis_prauc+avg_pro_prauc)/2, (avg_dis_p+avg_pro_p)/2, (avg_dis_r+avg_pro_r)/2, (avg_dis_f1+avg_pro_f1)/2
        avg_dis_cnt, avg_pro_cnt = dis_cnt / visit_cnt, pro_cnt / visit_cnt
        avg_cnt = (avg_dis_cnt+avg_pro_cnt)/2
    logging.info('Epoch {:03d}   Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_CNT: {:.4}'.format(
        epoch, avg_ja, avg_prauc, avg_p, avg_r, avg_f1, avg_cnt))
    logging.info('Epoch {:03d}   DISEASE Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_DIS_CNT: {:.4}'.format(
        epoch, avg_dis_ja, avg_dis_prauc, avg_dis_p, avg_dis_r, avg_dis_f1, avg_dis_cnt))
    logging.info('Epoch {:03d}   PROCEDURE Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_PRO_CNT: {:.4}'.format(
        epoch, avg_pro_ja, avg_pro_prauc, avg_pro_p, avg_pro_r, avg_pro_f1, avg_pro_cnt))
    return loss_val/len_val, avg_ja, avg_prauc, avg_p, avg_r, avg_f1, avg_cnt

@torch.no_grad()
def evaluator_nsp(model, data_val, data, epoch, device, mode='pretrain_nsp'):
    model.eval()
    loss_val = 0
    prc = []
    for batch_data in tqdm(data_val, ncols=60, desc=mode, total=len(data_val)):
        nsp_batch, nsp_target = nsp_batch_data(batch_data, data, neg_sample_rate=1)
        result = model(nsp_batch, mode='pretrain_nsp')
        loss = F.cross_entropy(result, torch.tensor(nsp_target, device=device, dtype=torch.float32))
        
        loss_val += loss.item()
        result = result.cpu().numpy()
        
        prc.append(np.mean((result>0.5)==nsp_target))
    return np.mean(prc), loss_val


def random_mask_word(seq, vocab, mask_prob=0.15):
    mask_idx = vocab.word2idx['[MASK]']
    for i, _ in enumerate(seq):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob: # 这个比例或许可以改一下
            prob /= mask_prob
            # 80% randomly change token to mask token
            if prob < 0.8:
                seq[i] = mask_idx
            # 10% randomly change token to random token
            elif prob < 0.9:
                seq[i] = random.choice(list(vocab.word2idx.items()))[1]
            else:
                pass
        else:
            pass
    return seq

# mask batch data
def mask_batch_data(batch_data, diag_voc, pro_voc, mask_prob):
    masked_data = []
    for visit in batch_data:
        diag = random_mask_word(visit[0], diag_voc, mask_prob)
        pro = random_mask_word(visit[1], pro_voc, mask_prob)
        masked_data.append([diag, pro])
    return masked_data

def nsp_batch_data(batch_data, data, neg_sample_rate=1):

    weights = [item[3] for item in data]
    nsp_batch = []
    nsp_target = []
    for visit in batch_data:
        nsp_batch.append(visit)
        nsp_target.append(1)
        for i in range(neg_sample_rate):
            neg_visit = random.choice(data)#, weights=weights, k=1)[0]
            while neg_visit[1] == visit[1]:
                neg_visit = random.choice(data)#, weights=weights, k=1)[0]
            if random.random() < 0.5:
                nsp_batch.append([visit[0], neg_visit[1]])
                nsp_target.append(0)
            else:
                nsp_batch.append([neg_visit[0], visit[1]])
                nsp_target.append(0)
    return nsp_batch, nsp_target


def main_mask(args, model, optimizer, writer, diag_voc, pro_voc, data_train, data_val, voc_size, device, save_dir, log_save_id):
    epoch_mask = 0
    best_epoch_mask, best_ja_mask = 0, 0
    EPOCH = args.pretrain_epochs
    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, logger={log_save_id}, mode=pretrain_mask')
        
        # mask pretrain
        model.train()
        epoch_mask += 1
        loss_train = 0
        for batch in tqdm(data_train, ncols=60, desc="pretrain_mask", total=len(data_train)):
            batch_size = len(batch)
            if args.mask_prob > 0:
                masked_batch = mask_batch_data(batch, diag_voc, pro_voc, args.mask_prob)
            else:
                masked_batch = batch
            result = model(masked_batch, mode='pretrain_mask')
            bce_target_dis = np.zeros((batch_size, voc_size[0]))
            bce_target_pro = np.zeros((batch_size, voc_size[1]))
            
            for i in range(batch_size):
                bce_target_dis[i, batch[i][0]] = 1
                bce_target_pro[i, batch[i][1]] = 1
            bce_target = np.concatenate((bce_target_dis, bce_target_pro), axis=1)
            
            # multi label margin loss
            multi_target_dis = np.full((1, voc_size[0]), -1)
            multi_target_pro = np.full((1, voc_size[1]), -1)
            for i in range(batch_size):
                multi_target_dis[i, 0:len(batch[i][0])] = batch[i][0]
                multi_target_pro[i, 0:len(batch[i][1])] = batch[i][1]
            multi_target = np.concatenate((multi_target_dis, multi_target_pro), axis=1)

            loss_bce = F.binary_cross_entropy_with_logits(result, torch.tensor(bce_target, device=device))
            loss_multi = F.multilabel_margin_loss(result, torch.tensor(multi_target, device=device))
            # loss = (1 - args.weight_multi) * loss_bce + args.weight_multi * loss_multi
            loss = loss_bce
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
        loss_train /= len(data_train)
        # validation
        loss_val, ja, prauc, avg_p, avg_r, avg_f1, avg_cnt = evaluator_mask(model, data_val, voc_size, epoch, device, mode='pretrain_mask')

        if ja > best_ja_mask:
            best_epoch_mask, best_ja_mask = epoch, ja
        logging.info(f'Training Loss_mask: {loss_train:.4f}, Validation Loss_mask: {loss_val:.4f}, best_ja: {best_ja_mask:.4f} at epoch {best_epoch_mask}\n')
        tensorboard_write_mask(writer, loss_train, loss_val, ja, prauc, epoch_mask)
    save_pretrained_model(model, save_dir)

def main_nsp(args, model, optimizer, writer, data, data_train, data_val, device, save_dir, log_save_id):
    
    epoch_nsp = 0
    best_epoch_nsp, best_prc_nsp = 0, 0
    EPOCH = args.pretrain_epochs
    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} -------------model_name={args.model_name}, logger={log_save_id}, mode=pretrain_nsp')

        model.train()
        epoch_nsp += 1
        loss_train = 0
        for batch in tqdm(data_train, ncols=60, desc="pretrain_nsp", total=len(data_train)):  # every patient
            nsp_batch, nsp_target = nsp_batch_data(batch, data, neg_sample_rate=1)
            result = model(nsp_batch, mode='pretrain_nsp')
            loss = F.cross_entropy(result, torch.tensor(nsp_target, device=device, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
        loss_train /= len(data_train)
        # validation
        precision, loss_val = evaluator_nsp(model, data_val, data, epoch, device, mode='pretrain_nsp')
        loss_val /= len(data_val)
        if precision > best_prc_nsp:
            best_epoch_nsp, best_prc_nsp = epoch, precision
        logging.info(f'Epoch {epoch:03d}   prc: {precision:.4}, best_prc: {best_prc_nsp:.4f} at epoch {best_epoch_nsp}, Training Loss_nsp: {loss_train:.4f}, Validation Loss_nsp: {loss_val:.4f}\n')
        tensorboard_write_nsp(writer, loss_train, loss_val, precision, epoch_nsp)
    save_pretrained_model(model, save_dir)

def save_pretrained_model(model, save_dir):
    # save the pretrained model
    model_path = os.path.join(save_dir, 'saved.pretrained_model')
    torch.save(model.state_dict(), open(model_path, 'wb'))
    logging.info('Pretrained model saved to {}'.format(model_path))

def load_pretrained_model(model, model_path):
    # load the pretrained model
    model.load_state_dict(torch.load(open(model_path, 'rb')))
    logging.info('Pretrained model loaded from {}'.format(model_path))

def loss_func(voc_size, patient, results, device):
    loss_bce_lst = loss_multi_lst = torch.Tensor().double().to(device) # modified
    for idx, adm in enumerate(patient):
        result = results[idx].unsqueeze(dim=0)
        loss_bce_target = np.zeros((1, voc_size[2]))  # (1, drug_num)
        loss_bce_target[:, adm[2]] = 1

        loss_multi_target = np.full((1, voc_size[2]), -1)
        loss_multi_target[0][0:len(adm[2])] = adm[2]

        loss_bce = F.binary_cross_entropy_with_logits(result, torch.tensor(loss_bce_target, device=device))        
        loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.tensor(loss_multi_target, device=device))   
        loss_bce_lst = torch.cat([loss_bce_lst, loss_bce.view(-1)])
        loss_multi_lst = torch.cat([loss_multi_lst, loss_multi.view(-1)])
    loss_bce_patient = loss_bce_lst.sum()
    loss_multi_patient = loss_multi.sum()
    return loss_bce_patient, loss_multi_patient

def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                      loss_train_bce=0.0, loss_train_multi=0.0, loss_train_ddi=0.0, loss_train_all=0.0,
                      loss_val_bce=0.0, loss_val_multi=0.0, loss_val_ddi=0.0, loss_val_all=0.0):
    if epoch > 0:
        writer.add_scalar('Loss_Train/bce', loss_train_bce, epoch)
        writer.add_scalar('Loss_Train/multi', loss_train_multi, epoch)
        writer.add_scalar('Loss_Train/ddi', loss_train_ddi, epoch)
        writer.add_scalar('Loss_Train/sum', loss_train_all, epoch)

        writer.add_scalar('Loss_Val/bce', loss_val_bce, epoch)
        writer.add_scalar('Loss_Val/multi', loss_val_multi, epoch)
        writer.add_scalar('Loss_Val/ddi', loss_val_ddi, epoch)
        writer.add_scalar('Loss_Val/sum', loss_val_all, epoch)

    writer.add_scalar('Metrics/Jaccard', ja, epoch)
    writer.add_scalar('Metrics/prauc', prauc, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate, epoch)
    writer.add_scalar('Metrics/Med_count', avg_med, epoch)

def tensorboard_write_mask(writer, loss_train, loss_val, ja, prauc, epoch):
    writer.add_scalar('Mask/Loss_Train_Mask', loss_train, epoch)
    writer.add_scalar('Mask/Loss_Val_Mask', loss_val, epoch)
    writer.add_scalar('Mask/Jaccard_Mask', ja, epoch)
    writer.add_scalar('Mask/prauc_Mask', prauc, epoch)
    
def tensorboard_write_nsp(writer, loss_train, loss_val, precision, epoch):
    writer.add_scalar('NSP/Loss_Train_NSP', loss_train, epoch)
    writer.add_scalar('NSP/Loss_Val_NSP', loss_val, epoch)
    writer.add_scalar('NSP/Precision_NSP', precision, epoch) 
    
def count_code_occurrences(records, voc_size):
    # 初始化总 vocab 长度的数组
    code_counts = np.zeros(voc_size[0] + voc_size[1], dtype=int)

    for adm in records:
        # 统计每个诊断 code 的出现次数
        for diag in adm[0]:
            code_counts[diag] += 1

        # 统计每个手术 code 的出现次数，注意偏移 voc_size[0] 以对齐索引
        for proc in adm[1]:
            code_counts[proc + voc_size[0]] += 1

    return code_counts

def relevance_mining(args, records, voc_size):
    file_path = f"../data/output/{args.dataset}/ehr_adj_ours.pkl"
    if os.path.exists(file_path):
        return dill.load(open(file_path, "rb"))

    # 药物-所有的ehr
    ehr_adj = np.zeros((voc_size[0] + voc_size[1], voc_size[0] + voc_size[1]))
    for adm in records:  # 遍历每次就诊记录
        for i in range(0, len(adm[0])):  # 遍历诊断
            for j in range(0, len(adm[1])):  # 诊断-手术
                node1 = adm[0][i]
                node2 = adm[1][j] + voc_size[0] - 1
                ehr_adj[node1, node2] += 1
            
            for k in range(i + 1, len(adm[0])):  # 诊断-诊断
                node1 = adm[0][i]
                node2 = adm[0][k]
                ehr_adj[node1, node2] += 1
                ehr_adj[node2, node1] += 1

        for i in range(len(adm[1])):  # 遍历手术
            for j in range(i + 1, len(adm[1])):  # 手术-手术
                node1 = adm[1][i] + voc_size[0] - 1
                node2 = adm[1][j] + voc_size[0] - 1
                ehr_adj[node1, node2] += 1
                ehr_adj[node2, node1] += 1

            for j in range(len(adm[0])):  # 诊断-手术
                node1 = adm[1][i] + voc_size[0] - 1
                node2 = adm[0][j] # + voc_size[0] - 1
                ehr_adj[node1, node2] += 1
                    
    return ehr_adj



def med_rel(args, records, voc_size):
    file_path = f"../data/output/{args.dataset}/ehr_adj_ours.pkl"
    if os.path.exists(file_path):
        return dill.load(open(file_path, "rb"))

    # 药物-所有的ehr
    med_adj = np.zeros((voc_size[2], voc_size[0] + voc_size[1]))
    for adm in records:
        for i in range(len(adm[2])):
            for j in range(len(adm[0])):
                node1 = adm[2][i]
                node2 = adm[0][j]
                med_adj[node1, node2] += 1
            for k in range(len(adm[1])):
                node1 = adm[2][i]
                node2 = adm[1][k]
                med_adj[node1, node2+voc_size[1]] += 1
                
    return med_adj