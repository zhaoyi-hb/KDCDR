import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from algorithms.GAlign.utils import *

from algorithms.GAlign.embedding_model import G_Align as Multi_Order
class KDCDR(nn.Module):

    def __init__(self, config, args):
        super(KDCDR, self).__init__()
        self.device = args.device
        self.latent_dim = args.embed_size  # int type: the embedding size of lightGCN
        self.n_layers = args.n_layers  # int type:  lightGCN层数
        self.reg_weight = args.regs  # float32 type: the weight decay for l2 normalization
        self.domain_lambda_source = args.lambda_s  # float32 type: the weight of source embedding in transfer function
        self.domain_lambda_target = args.lambda_t  # float32 type: the weight of target embedding in transfer function
        self.drop_rate = args.drop_rate  # float32 type: the dropout rate
        self.connect_way = args.connect_type  # str type: the connect way for all layers
        self.link_loss = args.link_loss

        self.source_num_users = config['n_users']
        self.target_num_users = config['n_users']
        self.source_num_items = config['n_items_s']
        self.target_num_items = config['n_items_t']
        self.n_fold = 1

        self.source_user_attr = torch.FloatTensor(config['user_att_s']).cuda()
        self.source_item_attr = torch.FloatTensor(config['item_att_s']).cuda()
        self.target_user_attr = torch.FloatTensor(config['user_att_t']).cuda()
        self.target_item_attr = torch.FloatTensor(config['item_att_t']).cuda()

        self.input_dim = args.input_dim
        self.latent_dim = 64


        self.n_interaction = args.n_interaction
        self.source_user_weight = torch.nn.Parameter(torch.empty(self.input_dim+self.latent_dim, self.latent_dim))
        self.source_item_weight = torch.nn.Parameter(torch.empty(self.input_dim+self.latent_dim, self.latent_dim))
        self.target_user_weight = torch.nn.Parameter(torch.empty(self.input_dim+self.latent_dim, self.latent_dim))
        self.target_item_weight = torch.nn.Parameter(torch.empty(self.input_dim+self.latent_dim, self.latent_dim))

        
        self.source_user_embedding = torch.nn.Parameter(torch.empty(self.source_num_users, self.latent_dim))
        self.target_user_embedding = torch.nn.Parameter(torch.empty(self.target_num_users, self.latent_dim))

        self.source_item_embedding = torch.nn.Parameter(torch.empty(self.source_num_items, self.latent_dim))
        self.target_item_embedding = torch.nn.Parameter(torch.empty(self.target_num_items, self.latent_dim))

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.reg_loss = EmbLoss()

        self.norm_adj_s = config['norm_adj_s']  
        self.norm_adj_t = config['norm_adj_t'] 
        self.domain_laplace = config['domain_adj']

        self.R_user_s = config['R_user_s']
        self.R_user_t = config['R_user_t']
        self.R_user_s,_ = Laplacian_graph(self.R_user_s)
        self.R_user_t,_ = Laplacian_graph(self.R_user_t)
        self.R_user_s = self.R_user_s.cuda()
        self.R_user_t = self.R_user_t.cuda()

        self.R_item_s = config['R_item_s']
        self.R_item_t = config['R_item_t']
        self.R_item_s,_ = Laplacian_graph(self.R_item_s)
        self.R_item_t,_ = Laplacian_graph(self.R_item_t)
        self.R_item_s = self.R_item_s.cuda()
        self.R_item_t = self.R_item_t.cuda()

        self.R_user_s_zhipu = config['R_user_s_zhipu']
        self.R_user_t_zhipu = config['R_user_t_zhipu']
        self.R_user_s_zhipu,_ = Laplacian_graph(self.R_user_s_zhipu)
        self.R_user_t_zhipu,_ = Laplacian_graph(self.R_user_t_zhipu)
        self.R_user_s_zhipu = self.R_user_s_zhipu.cuda()
        self.R_user_t_zhipu = self.R_user_t_zhipu.cuda()

        self.R_item_s_zhipu = config['R_item_s_zhipu']
        self.R_item_t_zhipu = config['R_item_t_zhipu']
        self.R_item_s_zhipu,_ = Laplacian_graph(self.R_item_s_zhipu)
        self.R_item_t_zhipu,_ = Laplacian_graph(self.R_item_t_zhipu)
        self.R_item_s_zhipu = self.R_item_s_zhipu.cuda()
        self.R_item_t_zhipu = self.R_item_t_zhipu.cuda()

       
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        torch.nn.init.xavier_normal_(self.source_user_weight, gain=1)
        torch.nn.init.xavier_normal_(self.source_item_weight, gain=1)
        torch.nn.init.xavier_normal_(self.target_user_weight, gain=1)
        torch.nn.init.xavier_normal_(self.target_item_weight, gain=1)
        torch.nn.init.xavier_normal_(self.source_user_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.target_user_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.source_item_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.target_item_embedding, gain=1)

        self.User_Embedding_Model = Multi_Order(
            activate_function = 'tanh',
            num_GCN_blocks = args.galign_gcn_blocks,
            input_dim = self.input_dim,
            output_dim = self.latent_dim,
            num_source_nodes = len(self.R_user_s),
            num_target_nodes = len(self.R_user_s),
            source_feats = self.source_user_attr,
            target_feats = self.target_user_attr,    
        )

        self.Item_Embedding_Model = Multi_Order(
            activate_function = 'tanh',
            num_GCN_blocks = args.galign_gcn_blocks,
            input_dim = self.input_dim,
            output_dim = self.latent_dim,
            num_source_nodes = len(self.R_user_s),
            num_target_nodes = len(self.R_user_s),
            source_feats = self.source_user_attr,
            target_feats = self.target_user_attr,    
        )
 
    def linkpred_loss(self, embedding, A):
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
        pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss
   
    def get_R(self,R,user):
        new_R = R[user, :]  # 获取指定的行
        new_R = new_R[:, user]  
        return new_R
    
    def get_attr(self, flag='s'):
        if flag == 's':
            user_attr = self.User_Embedding_Model(self.R_user_s_zhipu,'s',self.source_user_attr)
            item_attr = self.Item_Embedding_Model(self.R_item_s_zhipu,'s',self.source_item_attr)
            user_attr1 = torch.cat([user_attr[0],user_attr[-1]],dim=1)
            item_attr1 = torch.cat([item_attr[0],item_attr[-1]],dim=1)
            user_attr1 = torch.matmul(user_attr1,self.source_user_weight)
            user_attr1 = torch.nn.ReLU()(user_attr1)
            item_attr1 = torch.matmul(item_attr1,self.source_item_weight)
            item_attr1 = torch.nn.ReLU()(item_attr1)

        else:
            user_attr = self.User_Embedding_Model(self.R_user_t_zhipu,'t',self.target_user_attr)
            item_attr = self.Item_Embedding_Model(self.R_item_t_zhipu,'t',self.target_item_attr)
            user_attr1 = torch.cat([user_attr[0],user_attr[-1]],dim=1)
            item_attr1 = torch.cat([item_attr[0],item_attr[-1]],dim=1)
            user_attr1 = torch.matmul(user_attr1,self.target_user_weight)
            user_attr1 = torch.nn.ReLU()(user_attr1)
            item_attr1 = torch.matmul(item_attr1,self.target_item_weight)
            item_attr1 = torch.nn.ReLU()(item_attr1)

        return user_attr1,item_attr1,user_attr[-1],item_attr[-1]
        
    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding
            item_embeddings = self.source_item_embedding
            norm_adj_matrix = self.norm_adj_s       
        else:
            user_embeddings = self.target_user_embedding
            item_embeddings = self.target_item_embedding
            norm_adj_matrix = self.norm_adj_t
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def _split_A_hat(self, X,n_items):
        A_fold_hat = []
        fold_len = (self.source_num_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.source_num_users + n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(scipy_sparse_mat_to_torch_sparse_tensor(X[start:end]))
        return A_fold_hat


    def graph_layer(self, adj_matrix, all_embeddings):
        side_embeddings = torch.sparse.mm(adj_matrix.cuda(), all_embeddings)
        new_embeddings = side_embeddings + torch.mul(all_embeddings, side_embeddings)
        new_embeddings = self.dropout(new_embeddings)
        return new_embeddings
    
    def forward(self):

        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')

        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        for k in range(self.n_layers):
            source_all_embeddings = self.graph_layer(source_norm_adj_matrix, source_all_embeddings)
            target_all_embeddings = self.graph_layer(target_norm_adj_matrix, target_all_embeddings)

            source_norm_embeddings = nn.functional.normalize(source_all_embeddings, p=2, dim=1)
            target_norm_embeddings = nn.functional.normalize(target_all_embeddings, p=2, dim=1)
            source_embeddings_list.append(source_norm_embeddings)
            target_embeddings_list.append(target_norm_embeddings)

        if self.connect_way == 'concat':
            source_lightgcn_all_embeddings = torch.cat(source_embeddings_list, 1)
            target_lightgcn_all_embeddings = torch.cat(target_embeddings_list, 1)
        elif self.connect_way == 'mean':
            source_lightgcn_all_embeddings = torch.stack(source_embeddings_list, dim=1)
            source_lightgcn_all_embeddings = torch.mean(source_lightgcn_all_embeddings, dim=1)
            target_lightgcn_all_embeddings = torch.stack(target_embeddings_list, dim=1)
            target_lightgcn_all_embeddings = torch.mean(target_lightgcn_all_embeddings, dim=1)

        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_lightgcn_all_embeddings,
                                                                   [self.source_num_users, self.source_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_lightgcn_all_embeddings,
                                                                   [self.target_num_users, self.target_num_items])
        
        user_attr1_s,item_attr1_s,self.user_attr_s,self.item_attr_s =self.get_attr()
        source_user_all_embeddings = torch.cat([source_user_all_embeddings,user_attr1_s],dim=1)
        source_item_all_embeddings = torch.cat([source_item_all_embeddings,item_attr1_s],dim=1)

        user_attr1_t,item_attr1_t,self.user_attr_t,self.item_attr_t =self.get_attr(flag='t')
        target_user_all_embeddings = torch.cat([target_user_all_embeddings,user_attr1_t],dim=1)
        target_item_all_embeddings = torch.cat([target_item_all_embeddings,item_attr1_t],dim=1)

        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings
    
    def calculate_single_loss(self, user, item, label, flag):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        
        if flag == "source":
            source_u_embeddings = source_user_all_embeddings[user]
            source_i_embeddings = source_item_all_embeddings[item]
            R_user_s = self.get_R(self.R_user_s,user)
            R_item_s = self.get_R(self.R_item_s,item)
        
            source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
            source_bce_loss = self.loss(source_output, torch.from_numpy(label).cuda().to(torch.float))
            source_reg_loss = self.reg_loss(source_u_embeddings, source_i_embeddings)
            source_loss = source_bce_loss + self.reg_weight * source_reg_loss

            link_loss = (self.linkpred_loss(self.user_attr_s[user],R_user_s) + self.linkpred_loss(self.item_attr_s[item],R_item_s)) / 2
            
            source_loss = self.link_loss*source_loss + (1-self.link_loss)*link_loss
            return source_loss, 0

        if flag == "target":
            target_u_embeddings = target_user_all_embeddings[user]
            target_i_embeddings = target_item_all_embeddings[item]
            R_user_t = self.get_R(self.R_user_t,user)
            R_item_t = self.get_R(self.R_item_t,item)
        
            target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
            target_bce_loss = self.loss(target_output, torch.from_numpy(label).cuda().to(torch.float))
            target_reg_loss = self.reg_loss(target_u_embeddings, target_i_embeddings)
            target_loss = target_bce_loss + self.reg_weight * target_reg_loss
    
            link_loss = (self.linkpred_loss(self.user_attr_t[user],R_user_t) + self.linkpred_loss(self.item_attr_t[item],R_item_t)) / 2
           
            target_loss = self.link_loss*target_loss + (1-self.link_loss)*link_loss
            return 0, target_loss
        
    def calculate_cross_loss(self, source_user, source_item, source_label, target_user, target_item, target_label):
        

        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        
        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        R_user_s = self.get_R(self.R_user_s,source_user)
        R_item_s = self.get_R(self.R_item_s,source_item)
       
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]
        R_user_t = self.get_R(self.R_user_t,target_user)
        R_item_t = self.get_R(self.R_item_t,target_item)

        #====================================================================================================
        # calculate BCE Loss in source domain
        source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
        source_bce_loss = self.loss(source_output, torch.from_numpy(source_label).cuda().to(torch.float)) 

        # calculate Reg Loss in source domain
        source_reg_loss = self.reg_loss(source_u_embeddings, source_i_embeddings)  

        source_loss = source_bce_loss + self.reg_weight * source_reg_loss  

        
        source_link_loss = (self.linkpred_loss(self.user_attr_s[source_user],R_user_s) + self.linkpred_loss(self.item_attr_s[source_item],R_item_s)) / 2  
        source_loss = self.link_loss*source_loss + (1-self.link_loss)*source_link_loss  

        #===================================================================================================
        # calculate BCE Loss in target domain
        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        target_bce_loss = self.loss(target_output, torch.from_numpy(target_label).cuda().to(torch.float))  

        # calculate Reg Loss in target domain
        target_reg_loss = self.reg_loss(target_u_embeddings, target_i_embeddings)  

        target_loss = target_bce_loss + self.reg_weight * target_reg_loss  
        
        target_link_loss = (self.linkpred_loss(self.user_attr_t[target_user],R_user_t) + self.linkpred_loss(self.item_attr_t[target_item],R_item_t)) / 2 
        target_loss = self.link_loss*target_loss + (1-self.link_loss)*target_link_loss  
        
        losses = source_loss + target_loss

        return source_loss, target_loss, losses

    def predict(self, user, item, flag):
        if flag =="target":
            _, _, target_user_all_embeddings, target_item_all_embeddings = self.forward()
            u_embeddings = target_user_all_embeddings[user]
            i_embeddings = target_item_all_embeddings[item]
        else:
            source_user_all_embeddings, source_item_all_embeddings, _, _ = self.forward()
            u_embeddings = source_user_all_embeddings[user]
            i_embeddings = source_item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, user):
        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e

  

class EmbLoss(nn.Module):
    """
        EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
