import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn import LayerNorm
from torch.autograd import Variable
import math
from .gnn import GNNGraph
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, GlobalAttention, Set2Set

class GCNSet2Set(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNSet2Set, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dp = nn.Dropout(0.1)
        # self.set2set = Set2Set(hidden_dim, processing_steps=2, num_layers=1)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Set2Set输出为2 * hidden_dim
        self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)
            ))
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # 图卷积层
        x = self.conv1(x, edge_index, edge_weight)
        x = self.dp(F.relu(x))
        x = self.conv2(x, edge_index, edge_weight)
        x = self.dp(F.relu(x))
        # Set2Set池化
        x = self.pool(x, data.batch)  # batch用于支持多图，如果只有一个图，data.batch是全0
        # 全连接层映射到指定维度
        x = self.fc(x)
        return x
    

class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, projection_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1,
                                                                                2)  # [batch_size,num_classes, seq_len]
        weighted_output = att_weights @ x  # [batch_size,num_classes, input_size]
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class CausaltyReview(nn.Module):
    def __init__(self, casual_graph, num_diag, num_proc, num_med):
        super(CausaltyReview, self).__init__()

        self.num_med = num_med
        self.c1 = casual_graph

        # 直接用 Tensor 保存高低因果阈值
        self.register_buffer('c1_high_limit', torch.tensor(casual_graph.get_threshold_effect(0.97, "Diag", "Med")))
        self.register_buffer('c1_low_limit', torch.tensor(casual_graph.get_threshold_effect(0.90, "Diag", "Med")))

        # 权重参数，用于动态调整校正强度
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.01))  # 用于低条件校正
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.01))   # 用于高条件校正

    def forward(self, pre_prob, diags, procs):
        # 克隆原始概率
        reviewed_prob = pre_prob.clone()

        # 预计算所有 Diag 和 Med 的关系矩阵
        diag_med_effects = self.c1.get_effect_batch(diags, list(range(self.num_med)), "Diag", "Med")#.to(pre_prob.device)
        proc_med_effects = self.c1.get_effect_batch(procs, list(range(self.num_med)), "Proc", "Med")#.to(pre_prob.device)

        # 获取最大值
        max_cdm = torch.tensor(diag_med_effects.max(axis=0), device=pre_prob.device)  # shape: (num_med,)
        max_cpm = torch.tensor(proc_med_effects.max(axis=0), device=pre_prob.device)  # shape: (num_med,)

        # **动态校正机制：使用 Sigmoid 函数平滑调整幅度**
        low_condition = (max_cdm < self.c1_low_limit) & (max_cpm < self.c1_low_limit)
        high_condition = (max_cdm > self.c1_high_limit) | (max_cpm > self.c1_high_limit)

        # 动态权重调整：根据与阈值的差距决定校正力度
        low_adjustment = (1 + torch.sigmoid(self.c1_low_limit - torch.maximum(max_cdm, max_cpm))) * 0.5
        high_adjustment = (1 + torch.sigmoid(torch.maximum(max_cdm, max_cpm) - self.c1_high_limit)) * 0.5

        # 应用校正（批量化处理）
        reviewed_prob[0, low_condition] -= self.c1_minus_weight * low_adjustment[low_condition]
        reviewed_prob[0, high_condition] += self.c1_plus_weight * high_adjustment[high_condition]

        return reviewed_prob


class MAB(torch.nn.Module):
    def __init__(
        self, Qdim, Kdim, Vdim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(MAB, self).__init__(*args, **kwargs)
        self.Vdim = Vdim
        self.number_heads = number_heads

        assert self.Vdim % self.number_heads == 0, \
            'the dim of features should be divisible by number_heads'

        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)

        self.use_ln = use_ln
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)
            self.ln2 = torch.nn.LayerNorm(self.Vdim)

    def forward(self, X, Y):
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads

        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)
        O = Q_split + torch.matmul(Attn, V_split)
        O = torch.cat(O.split(batch_size, 0), 2)

        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)
        O = O if not self.use_ln else self.ln2(O)

        return O


class SAB(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(SAB, self).__init__(*args, **kwargs)
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X):
        return self.net(X, X)

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)

        Attn = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)

        fix_feat = torch.diag(fix_feat.squeeze(0))
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)
        return torch.sum(O, dim=1, keepdim=True)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).int().unsqueeze(0)
        x = x + self.embeddings(pos).expand_as(x)
        return x
    
class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe *= 0.1
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PatientEncoder(nn.Module): # 把patient_encoder抽象出来，方便后续的模型调用 
    def __init__(self, args, rare_diseases, rare_procedure, ehr_adj, causal_graph, voc_size, molecule_para, substruct_para, substruct_num):
        super(PatientEncoder, self).__init__()
        self.args = args
        self.voc_size = voc_size
        self.emb_dim = args.embed_dim  # 如果有预训练参数，保持与预训练的维度一致
        self.device = torch.device('cuda:{}'.format(args.cuda))
        self.tensor_ehr_adj = nn.Parameter(torch.FloatTensor(ehr_adj).to(self.device))
        self.graph = GCNSet2Set(input_dim=self.emb_dim, hidden_dim=256, output_dim=self.emb_dim)
        self.causal_graph = causal_graph
        self.positional_embedding_layer_graph = LearnablePositionalEncoding(d_model=self.emb_dim)

        self.rare_diseases = rare_diseases
        self.rare_procedures = rare_procedure
        self.rare_emb_d = nn.Embedding(len(self.rare_diseases), self.emb_dim)
        self.rare_emb_p = nn.Embedding(len(self.rare_procedures), self.emb_dim)
        # 稀有代码的动态增强向量
        self.rare_aug_d = nn.Parameter(torch.zeros(len(rare_diseases), self.emb_dim, device=self.device))
        self.rare_aug_p = nn.Parameter(torch.zeros(len(rare_procedure), self.emb_dim, device=self.device))
        torch.nn.init.xavier_uniform_(self.rare_aug_d)
        torch.nn.init.xavier_uniform_(self.rare_aug_p)

        # Attention mechanism to merge main and rare subspaces

        # self.embeddings = nn.ModuleList(
        #     [nn.Embedding(voc_size[i], self.emb_dim) for i in range(2)])  # 疾病 手术 
        
        # self.special_embeddings = nn.Embedding(2, self.emb_dim)                # 增加两个token：[CLS]， [SEP]
        self.special_tokens = {'CLS': torch.LongTensor([0,]).to(self.device), 'SEP': torch.LongTensor([1,]).to(self.device)}
        
        self.segment_embedding = nn.Embedding(2, self.emb_dim)
        self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, 128)
            )        
        self.global_encoder = GNNGraph(**molecule_para)

        self.sab = SAB(128, 128, 2, use_ln=True)
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )
        self.substruct_rela = LabelAttention(self.emb_dim, 256, substruct_num)
        self.aggregator = AdjAttenAgger(
           128, 128, max(128, 128)
        )
        score_extractor = [
            torch.nn.Linear(self.emb_dim, self.emb_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        # self.positional_embedding_layer = PositionalEncoding(d_model=args.embed_dim)
        # self.positional_embedding_layer = LearnablePositionalEncoding(d_model=args.embed_dim)

        if args.patient_seperate == False:
            self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], self.emb_dim) for i in range(2)])  # 疾病 手术 
            self.special_embeddings = nn.Embedding(3, self.emb_dim)                # 增加两个token：[CLS]， [SEP]
            self.transformer_visit = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )
            self.positional_embedding_layer_disease = LearnablePositionalEncoding(d_model=args.embed_dim)
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(d_model=args.embed_dim)
            self.patient_encoder = self.patient_encoder_unified
        else:
            self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], self.emb_dim//2) for i in range(2)])  # 疾病 手术 
            self.special_embeddings = nn.Embedding(2, self.emb_dim//2)                # 增加两个token：[CLS]， [SEP]
            self.transformer_disease = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_dim//2, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )
            self.transformer_procedure = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.emb_dim//2, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )

            self.patient_layer = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.ReLU(),
                nn.Linear(self.emb_dim, self.emb_dim),
            )

            self.positional_embedding_layer_disease = LearnablePositionalEncoding(d_model=args.embed_dim//2)
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(d_model=args.embed_dim//2)

            self.patient_encoder = self.patient_encoder_seperate
        
    
    def drug_emb(self, repr, sub_emb, global_emb, ddi_mask_H):
        query = self.query(repr.unsqueeze(1))
        substruct_weight = torch.sigmoid(self.substruct_rela(query))

        molecule_embeddings = self.aggregator(
            global_emb, sub_emb,
            substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
        )
        return molecule_embeddings



    def rare_emb(self, emb_d, emb_p, diseases, procedures):
        # 处理稀有疾病的嵌入
        rare_mask_d = torch.isin(diseases, torch.tensor(self.rare_diseases, dtype=torch.long, device=self.device))
        if rare_mask_d.any():
            rare_input_d = diseases[rare_mask_d]
            rare_indices_d = torch.tensor(
                [self.rare_diseases.index(code.item()) for code in rare_input_d], dtype=torch.long, device=self.device)
            rare_embeds_d = self.rare_emb_d(rare_indices_d)
            emb_d[rare_mask_d] = emb_d[rare_mask_d] + rare_embeds_d
            rare_aug_d = self.rare_aug_d[rare_indices_d]

        # 稀有操作处理
        rare_mask_p = torch.isin(procedures, torch.tensor(self.rare_procedures, dtype=torch.long, device=self.device))
        if rare_mask_p.any():
            rare_input_p = procedures[rare_mask_p]
            rare_indices_p = torch.tensor(
                [self.rare_procedures.index(code.item()) for code in rare_input_p], dtype=torch.long, device=self.device)
            rare_embeds_p = self.rare_emb_p(rare_indices_p) 
            emb_p[rare_mask_p] = emb_p[rare_mask_p] + rare_embeds_p
            rare_aug_p = self.rare_aug_p[rare_indices_p]
        
        if rare_mask_p.any() and rare_mask_d.any():
            sim =  F.sigmoid(self.rare_emb_d(rare_indices_d) @ self.rare_emb_p(rare_indices_p).transpose(0,1))  ## log5 用的是 aug_d @ aud_p
            id_d = torch.argmax(sim, dim=1)
            id_p = torch.argmax(sim.T, dim=1)
            emb_d[rare_mask_d] = emb_d[rare_mask_d] + rare_aug_p[id_d]
            emb_p[rare_mask_p] = emb_p[rare_mask_p] + rare_aug_d[id_p]

        return emb_d, emb_p

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log() 

    def patient_encoder_unified(self, batch_visits, substruct_data, mol_data, average_projection, ddi_mask_H):
        batch_repr = []
        repr_d = []
        repr_p = []
        global_embeddings = self.global_encoder(**mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings).unsqueeze(0) # 1, 112, 128
        substruct_embeddings = self.sab(self.substruct_emb.unsqueeze(0))  # 1, N, 128
        for adm in batch_visits:
            # 对每次访问：
            diseases = adm[0]
            procedures = adm[1]
            index = diseases + [(self.voc_size[0] + i) for i in procedures]
            diseases = torch.LongTensor(diseases).unsqueeze(dim=1).to(self.device)
            procedures = torch.LongTensor(procedures).unsqueeze(dim=1).to(self.device)

            disease_embedding = self.embeddings[0](diseases) # (n, 1, dim)
            procedure_embedding = self.embeddings[1](procedures)  # (m, 1, dim)
            
            disease_embedding, procedure_embedding  = self.rare_emb(disease_embedding, procedure_embedding, diseases, procedures)


            
            index = torch.tensor(index, dtype=torch.long)
            submatrix = self.tensor_ehr_adj[index, :][:, index]

            # diag_med_effects = self.causal_graph.get_effect_batch(diseases, list(range(self.voc_size[2])), "Diag", "Med") #.to(pre_prob.device)
            # proc_med_effects = self.causal_graph.get_effect_batch(procedures, list(range(self.voc_size[2])), "Proc", "Med") #.to(pre_prob.device)
            # diag_med_effects = torch.tensor(diag_med_effects, device=self.device, dtype=torch.float32)
            # proc_med_effects = torch.tensor(proc_med_effects, device=self.device, dtype=torch.float32)
            # diag_med_diag = diag_med_effects @ diag_med_effects.transpose(0,1)
            # proc_med_proc = proc_med_effects @ proc_med_effects.transpose(0,1)
            # diag_med_proc = diag_med_effects @ proc_med_effects.transpose(0,1)

            # casual_matrix = torch.zeros_like(submatrix, device=self.device, dtype=torch.float32)
            # # 填充子矩阵
            # casual_matrix[:len(diseases), :len(diseases)] = diag_med_diag         # 填充 R_A
            # casual_matrix[:len(diseases), len(diseases):] = diag_med_proc        # 填充 R_AB
            # casual_matrix[len(diseases):, :len(diseases)] = diag_med_proc.T      # 填充 R_AB 的转置
            # casual_matrix[len(diseases):, len(diseases):] = proc_med_proc
 

            edge_index = torch.nonzero(submatrix, as_tuple=False).T
            graph_embedding = torch.concat([disease_embedding, procedure_embedding], dim=0).squeeze(1)
            edge_weight = submatrix [edge_index[0], edge_index[1]] 
            data = Data(x=graph_embedding, edge_index=edge_index, edge_weight=edge_weight, batch=torch.zeros(graph_embedding.shape[0], dtype=torch.long))
            graph_embedding = self.graph(data.to(self.device))#.unsqueeze(1)

            #graph_embedding = torch.sum(graph_embedding, dim=0, keepdim=True).unsqueeze(1)
            #disease_embedding = disease_embedding + graph_d.unsqueeze(1)
            #procedure_embedding = procedure_embedding + graph_p.unsqueeze(1)
            cls_embedding = self.special_embeddings(self.special_tokens['CLS']).unsqueeze(dim=1)
            sep_embedding = self.special_embeddings(self.special_tokens['SEP']).unsqueeze(dim=1)
            
            disease_embedding = torch.cat((cls_embedding, disease_embedding), dim=0)  # (n+1, 1, dim)
            procedure_embedding = torch.cat((sep_embedding, procedure_embedding), dim=0) # (m+1, 1, dim)

            disease_embedding = self.positional_embedding_layer_disease(disease_embedding)
            procedure_embedding = self.positional_embedding_layer_procedure(procedure_embedding)

            combined_embedding = torch.cat((disease_embedding, procedure_embedding), dim=0)  # (n+m+2, 1, dim)
            
            
            # 加入segment embedding
            segments = torch.tensor([0] * (len(diseases) + 2) + [1] * len(procedures)).to(self.device)
            segment_embedding = self.segment_embedding(segments).unsqueeze(dim=1)
            input_embedding = combined_embedding + segment_embedding

            visit_representation = self.transformer_visit(input_embedding)


            disease_embedding = visit_representation[1:len(diseases)]
            procedure_embedding = visit_representation[1+len(diseases):]

            repr_d.append(torch.sum(disease_embedding, dim=0))
            repr_p.append(torch.sum(procedure_embedding, dim=0))

            visit_representation = visit_representation[0] + graph_embedding #torch.concat([visit_representation[0],graph_embedding.squeeze(1)], dim=-1)

            drug_embed = self.drug_emb(visit_representation, substruct_embeddings, global_embeddings, ddi_mask_H.to(self.device))
            visit_representation = torch.reshape(visit_representation, (1,1,-1)) 
            visit_representation = torch.concat([visit_representation, drug_embed], dim=-1)  # (1,1,dim)
            batch_repr.append(visit_representation)
        batch_repr = torch.cat(batch_repr, dim=1).to(self.device)  # (1,B,dim)
        batch_repr = batch_repr.squeeze(dim=0)  # (B,dim)
        repr_d = torch.cat(repr_d)
        repr_p = torch.cat(repr_p)
        return batch_repr, repr_d, repr_p


class RAREMed(PatientEncoder):
    def __init__(self, args, rare_diseases, rare_procedure, ehr_adj, causal_graph, voc_size, ddi_adj, molecule_para, substruct_para, substruct_num):
        super(RAREMed, self).__init__(args, rare_diseases, rare_procedure, ehr_adj, causal_graph, voc_size, molecule_para, substruct_para, substruct_num)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(self.device)

        # self.patient_layer = Adapter(self.emb_dim, args.adapter_dim)

        self.init_weights()
        self.causal_graph = causal_graph
        # self.mask_adapter = Adapter(self.emb_dim, args.adapter_dim)
        self.cls_mask = LabelAttention(self.emb_dim+128, 256, self.voc_size[0]+self.voc_size[1])#nn.Linear(self.emb_dim+128, self.voc_size[0]+self.voc_size[1])
        self.review = CausaltyReview(causal_graph, voc_size[0], voc_size[1], voc_size[2])

        # self.nsp_adapter = Adapter(self.emb_dim, args.adapter_dim)
        self.cls_nsp =  LabelAttention(self.emb_dim+128, 256, 4)#nn.Linear(self.emb_dim+128, 1)

        self.cls_final =  LabelAttention(self.emb_dim+128, 256, self.voc_size[2])#nn.Linear(self.emb_dim+128, self.voc_size[2])
    
    def forward_finetune(self, input, substruct_data, mol_data, average_projection, ddi_mask_H):
        # B: batch size, D: drug num, dim: embedding dimension
        patient_repr,_, _ = self.patient_encoder(input, substruct_data, mol_data, average_projection, ddi_mask_H) # (B,dim)
        # patient_repr = self.patient_layer(patient_repr)  # (B,dim)

        result = self.cls_final(patient_repr.unsqueeze(1))  # (B,D)
        for i in input:
            result = self.review(result, i[0], i[1])

        neg_pred_prob = F.sigmoid(result)   # (B,D)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        # prompt the probability of indication medications to be higher
        # indication_medication = input[-1][4]
        # indication_medication = torch.LongTensor(indication_medication).to(self.device)
        # contraindication_medication = input[-1][5]
        # contraindication_medication = torch.LongTensor(contraindication_medication).to(self.device)

        # output_mean_0 = result.mean(dim=1)
        # result[0, indication_medication] += 2
        # result = result - result.mean(dim=1) + output_mean_0

        # output_mean_0 = result.mean(dim=1)
        # result[0, contraindication_medication] -= 0.1
        # result = result - result.mean(dim=1) + output_mean_0

        return result, batch_neg

    def forward(self, input, substruct_data, mol_data, ddi_mask_H, tensor_ddi_adj, average_projection, mode='fine-tune'):
        assert mode in ['fine-tune', 'pretrain_mask', 'pretrain_nsp']
        if mode == 'fine-tune':
            result, batch_neg = self.forward_finetune(input, substruct_data, mol_data, average_projection, ddi_mask_H)
            return result, batch_neg
        
        elif mode == 'pretrain_mask':
            patient_repr,_, _ = self.patient_encoder(input, substruct_data, mol_data, average_projection, ddi_mask_H)      # (B, dim)
            # patient_repr = self.mask_adapter(patient_repr)  # (B, dim)
            result = self.cls_mask(patient_repr.unsqueeze(1))            # (B, voc_size[0]+voc_size[1])
            return result
        
        elif mode == 'pretrain_nsp':
            patient_repr, repr_d, repr_p = self.patient_encoder(input, substruct_data, mol_data, average_projection, ddi_mask_H)      # (B, dim)
            # patient_repr = self.nsp_adapter(patient_repr)   # (B, dim)
            result = self.cls_nsp(patient_repr.unsqueeze(1))             # (B, 1)
            result = result.squeeze(dim=1)                  # (B,)
            logit = F.sigmoid(result) # logistic regression
            return logit, repr_d, repr_p

    def init_weights(self):
        """Initialize embedding weights."""
        initrange = 0.1
        self.embeddings[0].weight.data.uniform_(-initrange, initrange)      # disease
        self.embeddings[1].weight.data.uniform_(-initrange, initrange)      # procedure
        self.rare_emb_d.weight.data.uniform_(-initrange, initrange)
        self.rare_emb_p.weight.data.uniform_(-initrange, initrange)
        self.segment_embedding.weight.data.uniform_(-initrange, initrange)
        self.special_embeddings.weight.data.uniform_(-initrange, initrange)
