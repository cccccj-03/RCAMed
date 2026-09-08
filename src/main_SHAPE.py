import collections
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
import torch.optim as optim
from torch.utils import data
# from loss import cross_entropy_loss
import os
import torch.nn.functional as F
import random
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from utils.data_loader2 import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
from utils.util import llprint, get_n_params, output_flatten, create_log_id, logging_config, get_model_path
import logging
from torch.utils.tensorboard import SummaryWriter

import sys

# sys.path.append("..")
from models.shape_model import SHAPE
from utils.utils_shape import llprint, sequence_output_process, ddi_rate_score, get_n_params, \
    print_result
from utils.recommend2 import eval, test
from copy import deepcopy

# torch.manual_seed(1203)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model_name = 'Set_GMed'
# resume_path = ''

# if not os.path.exists(os.path.join("saved", model_name)):
#     os.makedirs(os.path.join("saved", model_name))

"""adjust_learning_rate"""

# 计算衰减学习率
def lr_poly(base_lr, iter, max_iter, power, current_length):
    # ratio_length = 1 - (float(current_length) / 30)
    # iter = iter + ratio_length
    iter = iter + current_length
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))  # + (float(current_length) / 30) ** (power))
    # return base_lr * (((1 - float(iter) / max_iter) ** (power))+ 0.1*((1 - (float(current_length) / 30) ** (power))))

# 应用学习率调整优化器
def adjust_learning_rate(optimizer, i_iter, args, current_length):
    lr = lr_poly(args.lr, i_iter, args.num_steps, args.power, current_length)
    optimizer.param_groups[0]['lr'] = np.min(np.around(lr, 8))
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='1', help="User notes")
    parser.add_argument('-t', '--test', action='store_true',default=True, help="test mode")
    parser.add_argument('-s', '--single', action='store_true', default=False, help="single visit")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log0", help='log dir prefix like "log0"')
    parser.add_argument('--model_name', type=str, default="SHAPE", help="model name")
    parser.add_argument('--dataset', type=str, default="mimic-iii", help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')
    parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop after this many epochs without improvement')
    parser.add_argument('--cuda', type=int, default=5, help='which cuda')
    parser.add_argument('--embed_dim', type=int, default=128, help='dimension of node embedding(randomly initialize)')
    parser.add_argument('--ln', type=int, default=0, help='layer normlization')
    parser.add_argument('--threshold', type=float, default=0.3, help='the threshold of prediction')
    parser.add_argument('--kgloss', type=float, default=0.001, help='Choose GPU device')

    args = parser.parse_args()
    return args

def main(args):
    # set logger
    if args.test:
        args.note = f'test of {args.log_dir_prefix}'
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log' + str(log_save_id) + '_' + args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # load data
    data_path = f'/home/chenjian/mm_EHR/drugrec/RAREMed/data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'/home/chenjian/mm_EHR/drugrec/RAREMed/data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'/home/chenjian/mm_EHR/drugrec/RAREMed/data/output/{args.dataset}' + '/ddi_A_final.pkl'
    ehr_adj_path = f'/home/chenjian/mm_EHR/drugrec/RAREMed/data/output/{args.dataset}' + '/ehr_adj_final.pkl'
    ddi_mask_path = f'/home/chenjian/mm_EHR/drugrec/RAREMed/data/output/{args.dataset}' + '/ddi_mask_H.pkl'

    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1

    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x: med_count[x])
            data[i][j][2] = cur_medications

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    if args.single:
        data_train = [[visit] for patient in data_train for visit in patient]
        data_val = [[visit] for patient in data_eval for visit in patient]
        data_test = [[visit] for patient in data_test for visit in patient]

    # # sorted data according the visit length
    # def sorted_data(data):
    #     # data_len = [len(i) for i in data]
    #     data = sorted(data, key=lambda x:len(x))
    #     return data
    # data_train = sorted_data(data_train)
    # data_eval = sorted_data(data_eval)
    # data_test = sorted_data(data_test)

    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_eval)
    test_dataset = mimic_data(data_test)
    total_dataset = mimic_data(data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train,
                                  shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True,
                                 pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False,
                                 pin_memory=True)
    total_dataloader = DataLoader(total_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True,
                                  pin_memory=True)

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]


    model = SHAPE(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=args.embed_dim, device=device,
                      dim_hidden=args.embed_dim, ln=args.ln, kgloss_alpha=args.kgloss)

    if args.test:
        print(args.log_dir_prefix)
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb'), map_location='cpu'))
        model.to(device=device)

        eval(model, eval_dataloader, voc_size,  device, TOKENS, args, ddi_adj_path, save_dir + '/rec_results.pkl')
        return
    else:
        writer = SummaryWriter(save_dir)

    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(model.parameters(), lr=args.lr)  # , weight_decay=0.01
    print('parameters', get_n_params(model))

    args.power = 0.9
    args.num_steps = 50000  # 50000
    args.weight_decay = 0.0005
    args.momentum = 0.9
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 使用SGD作为优化function
    # optimizer = optim.SGD(model.parameters(),
    #                       args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 200
    cu_iter = 0
    for epoch in range(EPOCH):
        tic = time.time()
        epoch += 1
        print(f'\nepoch {epoch} --------------------------model_name={args.model_name}, lr={args.lr}, '
              f'batch_size={args.batch_size}, beam_size={args.beam_size}, max_med_len={args.max_len}, logger={log_save_id}')
        model.train()
        for idx, data in enumerate(train_dataloader):
            diseases, procedures, medications, seq_length, \
                d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, target_list = data

            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            # medications = medications.to(device)
            medications = pad_num_replace(medications, -1, MED_PAD_TOKEN).to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)

            """
            下面增加adjust_learning_rate的代码
            """
            cu_iter += 1
            adjust_learning_rate(optimizer, cu_iter, args, np.sum(np.array(seq_length)) / args.batch_size)
            output_logits, kgddi_loss = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix,
                                              m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask,
                                              stay_disease_mask,
                                              dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)

            # 需要对输出部分pad部分处理掉
            # labels, predictions = output_flatten(target_list, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, max_len=args.max_len)
            bce_target = np.zeros([medications.shape[0], medications.shape[1], voc_size[2]])
            for b_i, med in enumerate(target_list):
                for v_i, m in enumerate(med):
                    bce_target[b_i, v_i, m] = 1
            labels = torch.Tensor(bce_target).to(device)
            loss = F.binary_cross_entropy_with_logits(output_logits, labels)
            loss += kgddi_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            llprint('\rtraining step: {} / {} loss:{:.4f} lr:{} sample_len:{}'.format(idx, len(train_dataloader),
                                                                                      loss.item(),
                                                                                      optimizer.param_groups[0]['lr'],
                                                                                      diseases.shape))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, voc_size, device,TOKENS, args, ddi_adj_path)
        # ddi_rate, ja, prauc, avg_f1, avg_med = eval(args, epoch, model, eval_dataloader, voc_size, ddi_adj_path)
        logging.info('training time: {:.1f}, test time: {:.1f}'.format(time.time() - tic, time.time() - tic2))
        tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                          loss)

        # save best epoch
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_prauc, best_ddi_rate, best_avg_med = ja, prauc, ddi_rate, avg_med
            best_model_state = deepcopy(model.state_dict())
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))
        # print ('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))

        if epoch - best_epoch > args.early_stop:  # n个epoch内，验证集性能不上升之后就停
            break

    # save best model
    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                                                       'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja,
                                                                                                  ddi_rate)), 'wb'))

def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                      loss_train=0):
    if epoch > 0:
        writer.add_scalar('Loss/Train', loss_train, epoch)

    writer.add_scalar('Metrics/Jaccard', ja, epoch)
    writer.add_scalar('Metrics/prauc', prauc, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate, epoch)
    writer.add_scalar('Metrics/Med_count', avg_med, epoch)



if __name__ == '__main__':
    torch.manual_seed(1203)
    np.random.seed(2048)
    args = get_args()
    main(args)