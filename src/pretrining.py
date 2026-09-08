import torch
import torch.nn.functional as F
import torch.nn 
import numpy as np
from sklearn.metrics import f1_score
from utils.data_loader import pad_batch_v2_pre, pad_num_replace
import random
from tqdm import tqdm
import os 
from utils.util import llprint, get_n_params, output_flatten, create_log_id, logging_config, get_model_path
import torch.nn as nn
import torch.nn.functional as F
import torch
torch.autograd.detect_anomaly(True)


import torch.nn.functional as F
import torch

def pre_batch_data(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, neg_sample_rate=1):
    # print(data[0])
    new_batch = [[], [], [], [], [], []]
    ori_batch = [diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix]
    target = []
    len_batch = diseases.shape[0]
    if len_batch == 1:
        return ori_batch, [0]
    # neg_patient = random.choices(data, weights=weight, k=1)
    for i in range(0, len_batch):
    
        index = random.randint(0, len_batch-1)
        while index == i:
            index = random.randint(0, len_batch-1)
        
        for j in range(0,6):    
            new_batch[j].append(ori_batch[j][i])
        target.append(0)

        for j in range(0,6):
            if j == 1 or j == 4:
                new_batch[j].append(ori_batch[j][index])
            else:
                new_batch[j].append(ori_batch[j][0])
        target.append(1)

        for j in range(0,6):
            if j == 0 or j == 3:
                new_batch[j].append(ori_batch[j][index])
            else:
                new_batch[j].append(ori_batch[j][0])
        target.append(1)

    new_batch = [torch.stack(i) for i in new_batch]
    return new_batch, target


@torch.no_grad()
def evaluator_pre(model, data_val, voc_size, device):
    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    # model.eval()
    loss_val = 0
    F1 = []
    alignment_cos_sim = nn.CosineSimilarity(dim=1)
    for batch_data in tqdm(data_val, ncols=60, desc='pretraing', total=len(data_val)):

        diseases, procedures, medications, visit_weights_patient, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
        
        diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
        procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
        medications = medications.to(device)
        m_mask_matrix = m_mask_matrix.to(device)
        d_mask_matrix = d_mask_matrix.to(device)
        p_mask_matrix = p_mask_matrix.to(device)
        
        data_new, target = pre_batch_data(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix \
                                                    , device)
        diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix = data_new

        repr_dis, repr_pro = model.pretrain(diseases, procedures, medications, \
                                d_mask_matrix, p_mask_matrix, m_mask_matrix)
        mask = torch.ones(repr_dis.shape[0]//3).to(device)
        
        loss = 0
        for i in range(0, repr_dis.shape[0], 3):
            if i + 2 < repr_dis.shape[0]:  # 确保有三个元素
                sim_01 = (alignment_cos_sim(repr_dis[i:i+1,:], repr_dis[i+2:i+3,:]).abs() * mask).sum() / max(mask.sum(), 1e-6)  # B[0] 与 B[1] 的相似度
                sim_02 = (alignment_cos_sim(repr_pro[i:i+1,:], repr_pro[i+1:i+2, :]).abs() * mask).sum() / max(mask.sum(), 1e-6)   # B[0] 与 B[2] 的相似度
                loss = loss + sim_01 + sim_02
        # loss = (alignment_cos_sim(repr_dis, repr_pro).abs() * mask).sum() / max(mask.sum(), 1e-6)
        #loss = F.cross_entropy(result, torch.tensor(target, device=device))#, dtype=torch.float32))
        
        loss_val += loss#.item()
        # result = result.cpu().numpy()
    return loss_val
    #     predictions  = torch.argmax(result, dim=1)
    #     predictions = predictions.cpu().numpy()
    #     f1_macro = f1_score(target, predictions, average='macro')  # 每个类同等权重
    #     F1.append(f1_macro)#np.mean((result>0.5)==nsp_target))
    # return np.mean(F1), loss_val



def pretraining(args, save_dir, log_save_id, logging, model, data_train, data_val, optimizer, epoch_num, device, voc_size, writer):
    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    epoch_pre = 0
    best_epoch_pre, best_loss_val = 0, 999999
    EPOCH = epoch_num
    alignment_cos_sim = nn.CosineSimilarity(dim=1)
    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} -------------model_name={args.model_name}, logger={log_save_id}, mode=pretraining')
        model.train()
        epoch_pre += 1
        loss_train = 0
        for idx, batch_data in tqdm(enumerate(data_train), ncols=60, desc="pretrain_mask", total=len(data_train)):
            # optimizer.zero_grad()
            diseases, procedures, medications, seq_length, \
                d_length_matrix, p_length_matrix, m_length_matrix, \
                    d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                        dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
            
            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            
            data_new, target = pre_batch_data(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix \
                                                        , device)
            diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix = data_new

            repr_dis, repr_pro = model.pretrain(diseases, procedures, medications, \
                                    d_mask_matrix, p_mask_matrix, m_mask_matrix)
            mask = torch.ones(repr_dis.shape[0]//3).to(device)
            
            loss = 0
            for i in range(0, repr_dis.shape[0], 3):
                if i + 2 < repr_dis.shape[0]:  # 确保有三个元素
                    sim_01 = (alignment_cos_sim(repr_dis[i:i+1,:], repr_dis[i+2:i+3,:]).abs() * mask).sum() / max(mask.sum(), 1e-6)  # B[0] 与 B[1] 的相似度
                    sim_02 = (alignment_cos_sim(repr_pro[i:i+1,:], repr_pro[i+1:i+2, :]).abs() * mask).sum() / max(mask.sum(), 1e-6)   # B[0] 与 B[2] 的相似度
                    loss = loss + sim_01 + sim_02
        # loss = (alignment_cos_sim(repr_dis, repr_
            # loss = F.cross_entropy(result, torch.tensor(target, device=device)) #, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            loss_train += loss.item()
            # llprint('\rtraining step: {} / {}'.format(idx, len(data_train)))
        loss_train /= len(data_train)
        loss_val = evaluator_pre(model, data_val, voc_size, device)
        loss_val /= len(data_val)
        if loss_val < best_loss_val:
           best_epoch_pre, best_loss_val = epoch, loss_val
        logging.info(f'Epoch {epoch:03d}  , best_f1: {best_loss_val:.4f} at epoch {best_epoch_pre}, Training Loss_nsp: {loss_train:.4f}, Validation Loss_nsp: {loss_val:.4f}\n')
        tensorboard_write_pre(writer, loss_train, loss_val, epoch_pre)
    save_pretrained_model(model, logging, save_dir)


def save_pretrained_model(model, logging, save_dir):
    # save the pretrained model
    model_path = os.path.join(save_dir, 'saved.pretrained_model')
    torch.save(model.state_dict(), open(model_path, 'wb'))
    logging.info('Pretrained model saved to {}'.format(model_path))

def tensorboard_write_pre(writer, loss_train, loss_val, epoch):
    writer.add_scalar('NSP/Loss_Train_pretrain', loss_train, epoch)
    writer.add_scalar('NSP/Loss_Val_pretrain', loss_val, epoch)
  #  writer.add_scalar('NSP/F1_pretrain', precision, epoch) 

