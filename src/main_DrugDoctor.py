import dill
import pickle
import random
import numpy as np
import argparse
import logging
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import sys
import torch
from copy import deepcopy
import time
from models.DrugDoctor import AIModel
from utils.util import llprint, patient_to_visit, sequence_metric_drug, ddi_rate_score_drug, graph_batch_from_smile, get_n_params,output_flatten, create_log_id, logging_config, get_model_path
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from utils.data_loader_drugdoctor import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
from torch.utils.tensorboard import SummaryWriter

# setting
model_name = "AIDrug"
# resume_path = "./saved/AIDrug/Epoch_22_TARGET_0.06_JA_0.527_DDI_0.07488.model"

if not os.path.exists(os.path.join("saved_1", model_name)):
    os.makedirs(os.path.join("saved_1", model_name))

# Training settings
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--note', type=str, default='1', help="User notes")
    parser.add_argument('--dataset', type=str, default="mimic-iv", help='dataset')
    parser.add_argument("--test", action="store_true", default=True, help="test mode")
    parser.add_argument("--model_name", type=str, default=model_name, help="model name")
    parser.add_argument("--resume_path", type=str, default='/home/chenjian/mm_EHR/drugrec/RAREMed/src/1/drugdoctor/mimic3/log18_1/Epoch_27_JA_0.5272_DDI_0.06095.model', help="resume path")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--epoch", type=int, default=200, help="training epoches")
    parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
    parser.add_argument("--kgloss_alpha", type=float, default=0.5, help="kgloss_alpha for ddi_loss")
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--emb_dim', type=int, default=112, help='embedding dimension size')
    parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
    parser.add_argument("--dim", type=int, default=64, help="dimension")
    parser.add_argument("--cuda", type=int, default=5, help="which cuda")
    parser.add_argument('--threshold', type=float, default=0.4, help='the threshold of prediction')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop after this many epochs without improvement')
    parser.add_argument('-l', '--log_dir_prefix', type=str, default="log11", help='log dir prefix like "log0"')

    args = parser.parse_args()
    return args


# evaluate
def eval(model, eval_dataloader, drug_data, voc_size, device, TOKENS, args, ddi_adj_path, rec_results_path=None):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]  # 每个visit都要在list里存一个值
    med_cnt, visit_cnt = 0, 0

    ja_visit = [[] for _ in range(5)]
    rec_results = []
    for idx, data in enumerate(eval_dataloader):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        diagnoses, procedures, medications, used_medic, used_diag, used_proc, med_true, used_med_true, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix = data
        diagnoses = pad_num_replace(diagnoses, -1, DIAG_PAD_TOKEN).to(device)
        procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
        medications = medications.to(device)
        m_mask_matrix = m_mask_matrix.to(device)
        d_mask_matrix = d_mask_matrix.to(device)
        p_mask_matrix = p_mask_matrix.to(device)

        output_logits, ddi_loss = model(diagnoses, procedures, used_medic, used_diag, used_proc, drug_data,
                                        d_mask_matrix, p_mask_matrix)

        visit_cnt += len(diagnoses)

        # med_true 就是batch-size,voc_size[2] ，就是真实标签，0-1
        for i in range(len(diagnoses)):
            y_gt = med_true[i]  # groud truth 表示正确的label   0-1序列

            current_pre = output_logits[i]  # 模型预测的数据
            prediction = torch.sigmoid(current_pre).cpu().detach().numpy()
            y_pred_prob = prediction  # 预测的每一个药物的概率，非0-1序列

            out_list = np.where(prediction > args.threshold)[0]  # out_list就是预测的药物的编码号，比如0号药物，19号药物
            y_pred_label = out_list  # 预测的结果药物，非0-1序列，是存储当前visit预测的药物编码
            med_cnt += len(y_pred_label)

            y_pred = np.zeros(voc_size[2])
            y_pred[y_pred_label] = 1  # 预测的结果标签    0-1序列

            smm_record.append(y_pred_label)  #####

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric_drug(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))

            ja.append(adm_ja)  #
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)

    if args.test:
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        # plot_path = rec_results_path + '/' + 'pred_prob.jpg'
        # plot_hist(all_pred_prob, plot_path)
        ja_result_file = rec_results_path + '/' + 'ja_result.pkl'
        dill.dump(ja_visit, open(ja_result_file, 'wb'))
        for i in range(5):
            logging.info(
                str(i + 1) + f'visit\t mean: {np.mean(ja_visit[i]):.4}, std: {np.std(ja_visit[i]):.4}, se: {np.std(ja_visit[i]) / np.sqrt(len(ja_visit[i])):.4}')


    # ddi rate
    ddi_rate = ddi_rate_score_drug(smm_record, path=ddi_adj_path)

    llprint(
        "DDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main(args):
    if args.test:
        args.note = f'test of {args.log_dir_prefix}'
    log_directory_path = os.path.join('../log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path, 'log' + str(log_save_id) + '_' + args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # load data
    data_path = f'../data/output/{args.dataset}' + '/records_final.pkl'
    voc_path = f'../data/output/{args.dataset}' + '/voc_final.pkl'
    ddi_adj_path = f'../data/output/{args.dataset}' + '/ddi_A_final.pkl'
    molecule_path = f'../data/output/{args.dataset}/atc3toSMILES.pkl'
    substruct_smile_path = f'../data/output/{args.dataset}/substructure_smiles.pkl'
    ddi_mask_path = f'../data/output/{args.dataset}/ddi_mask_H.pkl'


    device = torch.device('cuda:{}'.format(args.cuda))

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data_all = dill.load(open(data_path, "rb"))
    molecule = dill.load(
        open(molecule_path, "rb")
    )
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = (
        voc["diag_voc"],
        voc["pro_voc"],
        voc["med_voc"],
    )

    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word),
    )

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    with open(substruct_smile_path, 'rb') as Fin:
        substruct_smiles_list = dill.load(Fin)
    substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
    substruct_forward = {'batched_data': substruct_graphs.to(device)}
    substruct_para = {
        'num_layer': 3, 'emb_dim': voc_size[2], 'graph_pooling': 'mean',
        'drop_ratio': 0.7, 'gnn_type': 'gin', 'virtual_node': False
    }  # 模型初始化时用到
    drug_data = {
        'substruct_data': substruct_forward,
        'ddi_mask_H': ddi_mask_H,
        'tensor_ddi_adj': ddi_adj
    }  # 模型训练时用到

    # 这一段是按照 data为patient-level进行分割训练集
    data = data_all
    print("all-data-size:", len(data))
    # 划分训练，验证，测试
    split_point = int(len(data) * 2 / 3)  # 4233 训练集长度
    data_train = data[:split_point]  # 训练集   patient-level data
    eval_len = int(len(data[split_point:]) / 2)  # 测试=验证集长度 1058,1059
    data_test = data[split_point: split_point + eval_len]  # patient-level data
    data_eval = data[split_point + eval_len:]  # patient-level data

    ########################################## 创建 visit-level 的数据集
    data_train_v = patient_to_visit(data_train, voc_size)
    data_test_v = patient_to_visit(data_test, voc_size)
    data_eval_v = patient_to_visit(data_eval, voc_size)
    print("data_train-size:", len(data_train))
    print("data_test-size:", len(data_test))
    print("data_eval-size:", len(data_eval))
    ###############################################################################################

    train_dataset = mimic_data(data_train_v)
    eval_dataset = mimic_data(data_eval_v)
    test_dataset = mimic_data(data_test_v)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train,
                                  shuffle=False, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=pad_batch_v2_eval, shuffle=False,
                                 pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=pad_batch_v2_eval, shuffle=False,
                                 pin_memory=True)

    model = AIModel(
        voc_size,
        ddi_adj,
        emb_dim=voc_size[2],
        substruct_para=substruct_para,
        kgloss_alpha=args.kgloss_alpha,
        device=device,
    )
    print('parameters:', get_n_params(model))
    print(model)

    print("test:", args.test)
    ############################## testing stage ###############################
    if args.test:
        # inference stage
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        model.load_state_dict(torch.load(open(model_path, 'rb'), map_location='cpu'))  # load model
        model.to(device=device)
        tic = time.time()

        result = []
        for _ in range(10):
            tatal_patient = len(data_test)
            random.shuffle(data_test)
            sampling = data_test[:int(0.8 * tatal_patient)]  # 80% patients in data_test
            data_test_v = patient_to_visit(sampling, voc_size)
            test_dataset = mimic_data(data_test_v)
            test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=pad_batch_v2_eval, shuffle=False,
                                         pin_memory=True)
            with torch.set_grad_enabled(False):
                ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, test_dataloader, drug_data, voc_size,
                                                                          device, TOKENS, args, ddi_adj_path, save_dir + '/rec_results.pkl')
                result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)

        print('test time: {}'.format(time.time() - tic))
        return
    else:
        writer = SummaryWriter(save_dir)  # 自动生成log文件夹

    ############################## training stage ###############################
    model.to(device=device)
    logging.info(f'n_parameters:, {get_n_params(model)}')
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0
    EPOCH = args.epoch

    for epoch in range(EPOCH):
        tic = time.time()
        print("epoch {} --------------------------".format(epoch))

        model.train()

        for idx, data in enumerate(train_dataloader):
            diagnoses, procedures, medications, used_medic, used_diag, used_proc, med_true, used_med_true, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix = data

            diagnoses = pad_num_replace(diagnoses, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)

            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)

            output1, loss_ddi = model(diagnoses, procedures, used_medic, used_diag, used_proc, drug_data, d_mask_matrix,
                                      p_mask_matrix)

            loss_current = F.binary_cross_entropy_with_logits(output1, torch.tensor(med_true).to(device))

            loss = loss_current + loss_ddi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print()
        tic2 = time.time()

        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, drug_data, voc_size, device,
                                                                  TOKENS, args, ddi_adj_path)
        print('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        # save best epoch
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_prauc, best_ddi_rate, best_avg_med = ja, prauc, ddi_rate, avg_med
            best_model_state = deepcopy(model.state_dict())
        logging.info('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))
        # print ('best_epoch: {}, best_ja: {:.4f}'.format(best_epoch, best_ja))

        if epoch - best_epoch > args.early_stop:  # n个epoch内，验证集性能不上升之后就停
            break

        if epoch - best_epoch > 10:
            break
        # if epoch  > 25:
        #     break

    logging.info('Train finished')
    torch.save(best_model_state, open(os.path.join(save_dir, \
                                                   'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(best_epoch, best_ja,
                                                                                              ddi_rate)), 'wb'))


if __name__ == '__main__':
    torch.manual_seed(1203)
    np.random.seed(2048)
    args = get_args()
    main(args)