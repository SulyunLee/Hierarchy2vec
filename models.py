

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import statistics
import tqdm
import sklearn
import random
import math
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from utils import *
from prepare_input_func import *

class Benchmark_Classifier(nn.Module):
    '''
    Generates a classifier that predicts target using the input features with
    a neural network architecture
    '''
    def __init__(self, input_feature_dim, hidden_node):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_node = hidden_node
        self.Dropout = nn.Dropout(0)

        self.fc = nn.Linear(input_feature_dim, hidden_node, bias=True)
        self.output_fc = nn.Linear(hidden_node, 1, bias=True)

    def forward(self, input_f):
        x = F.relu(self.fc(self.Dropout(input_f)))
        output = self.output_fc(self.Dropout(x))

        return output

class Nonhier_NN(nn.Module):
    '''
    Non-hierarchically aggregate coach features using fully-connected NN.
    '''
    def __init__(self, input_feature_dim, team_emb_dim, team_feature_dim, drop_rate):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.team_emb_dim = team_emb_dim
        self.team_feature_dim = team_feature_dim

        self.Dropout = nn.Dropout(drop_rate)

        self.agg_fc = nn.Linear(input_feature_dim, team_emb_dim, bias=True)
        self.output_fc = nn.Linear(team_emb_dim + team_feature_dim, 1, bias=True)

    def forward(self, node_agg_features, team_features):
        x = node_agg_features
        x = F.relu(self.agg_fc(self.Dropout(x)))
        x_concat = torch.cat([x, team_features], dim=1)
        output = self.output_fc(self.Dropout(x_concat))

        return output

class Hier_NN(nn.Module):
    '''
    Hierarchycally aggregate coach features by averaging and feeding into fully-connected NN
    '''
    def __init__(self, input_feature_dim, team_emb_dim, team_feature_dim, drop_rate):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.team_emb_dim = team_emb_dim
        self.team_feature_dim = team_feature_dim

        self.Dropout = nn.Dropout(drop_rate)

        self.agg_fc1 = nn.Linear(input_feature_dim * 2, input_feature_dim, bias=True)
        self.agg_fc2 = nn.Linear(input_feature_dim * 3, team_emb_dim, bias=True)
        self.output_fc = nn.Linear(team_emb_dim + team_feature_dim, 1, bias=True)

    def forward(self, offensive_position, defensive_position, offensive_coord, defensive_coord, hc, team_features):

        offensive_combined = torch.cat([offensive_position, offensive_coord], dim=1)
        defensive_combined = torch.cat([defensive_position, defensive_coord], dim=1)
        
        offensive_emb = F.relu(self.agg_fc1(self.Dropout(offensive_combined)))
        defensive_emb = F.relu(self.agg_fc1(self.Dropout(defensive_combined)))

        off_def_hc_combined = torch.cat([offensive_emb, defensive_emb, hc], dim=1)
        team_emb = F.relu(self.agg_fc2(self.Dropout(off_def_hc_combined)))

        x_concat = torch.cat([team_emb, team_features], dim=1) 
        output = self.output_fc(self.Dropout(x_concat))

        return output


class Attention_Biased_Walk(nn.Module):
    def __init__(self, collab_f_dim, is_hierarchy, is_strength, is_recency, drop_rate):
        super().__init__()

        self.collab_f_dim = collab_f_dim
        self.is_hierarchy = is_hierarchy
        self.is_strength = is_strength
        self.is_recency = is_recency

        self.attn_fc1 = nn.Linear(self.collab_f_dim*2, 1, bias=False)
        self.attn_fc2 = nn.Linear(self.collab_f_dim*2, 2, bias=False)
        self.Dropout = nn.Dropout(drop_rate)

    def forward(self, hierarchy, strength, recency):
        H = torch.Tensor(hierarchy)
        S = torch.Tensor(strength)
        R = torch.Tensor(recency)

        # calculate the attention coefficients
        attn_coef_lst = []
        vector_lst = []

        if self.is_hierarchy:
            if self.is_strength:
                HS = torch.cat([H, S], dim=1)
                e_HS = F.leaky_relu(self.attn_fc1(self.Dropout(HS)))
                attn_coef_lst.append(e_HS)
                vector_lst.append(S)

            if self.is_recency:
                HR = torch.cat([H, R], dim=1)
                e_HR = F.leaky_relu(self.attn_fc1(self.Dropout(HR)))
                attn_coef_lst.append(e_HR)
                vector_lst.append(R)

            HH = torch.cat([H, H], dim=1)
            e_HH = F.leaky_relu(self.attn_fc1(self.Dropout(HH)))
            attn_coef_lst.append(e_HH)
            vector_lst.append(H)

            # softmax attention coeficients
            alpha = F.softmax(self.Dropout(torch.stack(attn_coef_lst, dim=2)), dim=2)
            v = torch.stack(vector_lst, dim=2)
            agg_collab_f = torch.sum(alpha * v, dim=2)  
        else:
            SR = torch.cat([S, R], dim=1)
            e_SR = F.leaky_relu(self.attn_fc2(self.Dropout(SR)))
            alpha = F.softmax(self.Dropout(torch.stack([e_SR[:,0].view(-1,1),\
                                                        e_SR[:,1].view(-1,1)], dim=2)), dim=2)
            v = torch.stack([S, R], dim=2)
            agg_collab_f = F.relu(torch.sum(alpha*v, dim=2))

        return alpha, agg_collab_f
    

class Nonhier_NN_BiasAttn(nn.Module):
    '''
    Non-hierarchically aggregate coach features using fully-connected NN.
    Also, collaboration features based on biase walks are aggregated using attention.
    '''
    def __init__(self, individual_features, collab_features, team_emb_dim, \
                        team_info_names, label_name, is_hierarchy, is_strength, \
                            is_recency, drop_rate):
        super().__init__()

        self.individual_features = individual_features
        self.collab_features = collab_features
        self.team_emb_dim = team_emb_dim
        self.team_info_names = team_info_names
        self.label_name = label_name
        self.is_hierarchy = is_hierarchy
        self.is_strength = is_strength
        self.is_recency = is_recency
        self.drop_rate = drop_rate

        self.total_f_dim = len(individual_features) + len(collab_features)
        
        self.attn_layer = Attention_Biased_Walk(len(collab_features), \
                                                is_hierarchy, is_strength, \
                                                is_recency, drop_rate)
        self.agg_fc = nn.Linear(self.total_f_dim, \
                team_emb_dim, bias=True)
        self.output_fc = nn.Linear(team_emb_dim, 1, bias=True)

        self.Dropout = nn.Dropout(self.drop_rate)

    def forward(self, individual_f, hierarchy_collab_f, strength_collab_f,\
                    recency_collab_f, team_info, labels):
        # aggregate biased walks using attention mechanism
        alpha, agg_collab_f = self.attn_layer(hierarchy_collab_f, strength_collab_f, \
                                    recency_collab_f)

        # Combine individual features, aggregated collaboration features, and labels
        individual_df = pd.DataFrame(individual_f.detach().numpy(), \
                                    columns=self.individual_features)
        agg_collab_df = pd.DataFrame(agg_collab_f.detach().numpy(),\
                                    columns=self.collab_features)
        team_info_df = pd.DataFrame(team_info, columns=self.team_info_names)

        record_df = pd.concat([team_info_df, individual_df, agg_collab_df], axis=1)

        # Average coach features in each team
        x = torch.Tensor(record_df.groupby(self.team_info_names).mean().values)
        labels = torch.Tensor(labels[self.label_name]).view(-1,1)

        # Generate team embedding
        x = F.relu(self.agg_fc(self.Dropout(x)))
        output = self.output_fc(self.Dropout(x))
        
        return labels, output, alpha

class Hier_NN_BiasAttn(nn.Module):
    '''
    Hierarchically aggregate coach features using fully-connected NN.
    Also, collaboration features based on biased walks are aggregated using attention.
    '''
    def __init__(self, indiv_f_dim, collab_f_dim, emb_dim, is_hierarchy, is_strength, \
                is_recency, drop_rate): 
        super().__init__()

        self.indiv_f_dim = indiv_f_dim
        self.collab_f_dim = collab_f_dim
        self.emb_dim = emb_dim
        self.is_hierarchy = is_hierarchy
        self.is_strength = is_strength
        self.is_recency = is_recency
        self.drop_rate = drop_rate

        self.total_f_dim = self.indiv_f_dim + self.collab_f_dim
        
        self.attn_layer = Attention_Biased_Walk(collab_f_dim, is_hierarchy,is_strength, \
                                               is_recency, drop_rate)
        self.agg_fc1 = nn.Linear(self.total_f_dim * 2, self.total_f_dim, bias=True)
        self.agg_fc2 = nn.Linear(self.total_f_dim * 3, self.emb_dim, bias=True)
        self.output_fc = nn.Linear(self.emb_dim, 1, bias=True)
        self.Dropout = nn.Dropout(self.drop_rate)

    def positionwise_average(self, x, pos_id_tensor, pos_id, season_id_tensor):
        idx = (pos_id_tensor == pos_id).nonzero()[:,0]
        pos_f_tensor = x[idx,:]

        season_ids = season_id_tensor[idx,:]
        M = torch.zeros(int(season_ids.max())+1, pos_f_tensor.shape[0])
        M[season_ids.view(-1), torch.arange(pos_f_tensor.shape[0], out=torch.LongTensor())] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        averaged_tensor = torch.mm(M, pos_f_tensor)

        return averaged_tensor
        

    def forward(self, indiv, hierarchy_f, strength_f, recency_f, season_ids, pos_ids):
        # aggregate biased walks using attention mechanism
        alpha, agg_collab_f = self.attn_layer(hierarchy_f, strength_f, \
                                   recency_f)
        # combine with individual features
        x = torch.cat([indiv, agg_collab_f], dim=1)

        o = self.positionwise_average(x, pos_ids, 1, season_ids.type(torch.LongTensor))
        d = self.positionwise_average(x, pos_ids, 2, season_ids.type(torch.LongTensor))
        oc = self.positionwise_average(x, pos_ids, 3, season_ids.type(torch.LongTensor))
        dc = self.positionwise_average(x, pos_ids, 4, season_ids.type(torch.LongTensor))
        hc = self.positionwise_average(x, pos_ids, 5, season_ids.type(torch.LongTensor))
        
        # Hierarchically combine coach features
        offensive_emb = self.agg_fc1(self.Dropout(torch.cat([o, oc], dim=1)))
        offensive_emb = F.relu(offensive_emb)
        defensive_emb = self.agg_fc1(self.Dropout(torch.cat([d, dc], dim=1)))
        defensive_emb = F.relu(defensive_emb)

        all_cat = torch.cat([offensive_emb, defensive_emb, hc], dim=1)
        team_emb = F.relu(self.agg_fc2(self.Dropout(all_cat)))

        output = self.output_fc(self.Dropout(team_emb))

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None):
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)

        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = self.attention(q, k, v, self.d_k, mask)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)
        return output

class Nonhier_Optmatch(nn.Module):
    '''
    Non-hierarchically aggregate coach features using Optmatch by Gong et al.
    '''
    def __init__(self, indiv_f_dim, collab_f_dim, emb_dim,  
                    is_hierarchy, is_strength, is_recency, drop_rate):
        super().__init__()

        self.indiv_f_dim = indiv_f_dim
        self.collab_f_dim = collab_f_dim
        self.emb_dim = emb_dim
        self.is_hierarchy = is_hierarchy
        self.is_strength = is_strength
        self.is_recency = is_recency
        self.drop_rate = drop_rate

        self.total_f_dim = self.indiv_f_dim + self.collab_f_dim
        
        self.attn_layer = Attention_Biased_Walk(collab_f_dim, is_hierarchy,\
                                            is_strength, is_recency, drop_rate)
        self.transformer = MultiHeadAttention(1, indiv_f_dim+collab_f_dim,\
                                        drop_rate)
        self.Dropout = nn.Dropout(self.drop_rate)

        self.fc = nn.Linear(indiv_f_dim+collab_f_dim, emb_dim, bias=True)
        self.output_fc = nn.Linear(self.emb_dim, 1, bias=True)

    def forward(self, indiv_f, hierarchy_f, strength_f, recency_f, season_ids):
        # aggregate biased walks using attention mechanism
        alpha, agg_collab_f = self.attn_layer(hierarchy_f, strength_f, \
                                    recency_f)
        x = torch.cat([indiv_f, agg_collab_f], dim=1)

        team_emb = torch.zeros(season_ids.unique().size(0), self.emb_dim)
        i = 0
        for season_id in season_ids.unique():
            idx = (season_ids == season_id).nonzero()[:,0]
            team_members = x[idx,:]

            attn_output = self.transformer(team_members, team_members, team_members)
            member_emb = F.relu(self.fc(self.Dropout(attn_output)))
            team_emb[i,:] = member_emb.mean(dim=0)
            i += 1

        output = self.output_fc(self.Dropout(team_emb))

        return output




