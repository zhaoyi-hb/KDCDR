import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utility.helper import *
import torch
import pickle
from zhipuai import ZhipuAI
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix, save_npz, load_npz

class Data(object):
    def __init__(self, path, batch_size,neg_num,threshold,flag='None'):
        self.neg_num = neg_num
        self.path = path
        self.batch_size = batch_size
        self.threshold = threshold
        train_file = path +'/train.txt'
        test_file = path + '/test.txt'


        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file, "r") as f:
            line = f.readline().strip('\n')
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                self.n_users = max(self.n_users, u)
                self.n_items = max(self.n_items, i)
                self.n_train += 1
                line = f.readline().strip('\n')

        self.n_items += 1
        self.n_users += 1

        self.negativeList = self.read_neg_file(path)
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.ratingList = []
        self.train_items, self.test_set = {}, {}
        
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n').split('\t')
                    user, item, rating = int(l[0]), int(l[1]), float(l[2])
                    if user in self.train_items.keys():
                        self.train_items[user].append(item)
                    else:
                        self.train_items[user] = [item]
                    if (rating > 0):
                        self.R[user, item] = 1.0
                        # self.R[uid][i] = 1

                line = f_test.readline().strip('\n')
                while line != None and line != "":
                    arr = line.split("\t")
                    user, item = int(arr[0]), int(arr[1])
                    if user in self.test_set.keys():
                        self.test_set[user].append(item)
                    else:
                        self.test_set[user] = [item]
                    self.ratingList.append([user, item])
                    self.n_test += 1
                    line = f_test.readline().strip('\n')

        save_npz(path +"/user_item_A.npz",  self.R.tocsr() ) 
        
        if flag == 'zhipu':
            self.item_embedding = np.load(path+'/'+'zhipu_item_embeddings.npy')
            self.user_embedding = self.get_zhipu_uer_embedding()
        else:
            self.item_embedding = np.load(path+'/'+'item_embeddings.npy')
            self.user_embedding = self.get_uer_embedding()
        
        self.user_A_zhipu = self.get_zhipu_user_A()
        self.item_A_zhipu = self.get_zhipu_item_A()
        self.user_A = self.get_user_A()
        self.item_A = self.get_item_A()

        
    def get_R_mat(self):
        return self.R
    
    #  bert embedding
    def get_uer_embedding(self):
        
        user_A_file = self.path + '/user_embeddings.npy'

        if os.path.exists(user_A_file):
            uer_embedding = np.load(user_A_file)
            return uer_embedding
        
        adj = self.get_R_mat()
        user_num = adj.shape[0]
        user_embedding = []
        for node in range(user_num):
            item_neighbors = np.nonzero(adj[[node],:])[-1].tolist()
            user_embed_i = self.item_embedding[item_neighbors]
            user_embedding.append(np.mean(user_embed_i, axis=0))
        
        np.save(user_A_file, user_embedding)

        return user_embedding
    
    # 智谱embedding
    def get_zhipu_uer_embedding(self):
        
        user_A_file = self.path + '/zhipu_user_embeddings.npy'

        if os.path.exists(user_A_file):
            uer_embedding = np.load(user_A_file)
            return uer_embedding
        
        adj = self.get_R_mat()
        user_num = adj.shape[0]
        user_embedding = []
        for node in range(user_num):
            item_neighbors = np.nonzero(adj[[node],:])[-1].tolist()
            user_embed_i = self.item_embedding[item_neighbors]
            user_embedding.append(np.mean(user_embed_i, axis=0))
        
        np.save(user_A_file, user_embedding)

        return user_embedding

    def get_item_A(self):
        user_A_file = self.path + '/item_A.npy'

        if os.path.exists(user_A_file):
            user_A = np.load(user_A_file)
            return user_A
        
        adj = self.get_R_mat()
        item_num = adj.shape[1]
        user_A = np.zeros((item_num,item_num))
        for node in range(item_num):
            user_neighbors = set()
            item_neighbors = np.nonzero(adj[:,[node]])[0].tolist()  # 获取邻居节点
            for neighbor in item_neighbors:
                users = np.nonzero(adj[[neighbor],:])[1].tolist()
                if len(users) == 0:
                    continue
                user_neighbors = user_neighbors.union(users)
            user_A[node,np.array(list(user_neighbors)).astype(int)] = 1
            print("第{}为项目的一阶邻居是{}".format(node, user_neighbors))

        np.save(user_A_file, user_A)
        return user_A
    
    def get_user_A(self):
        user_A_file = self.path + '/user_A.npy'

        if os.path.exists(user_A_file):
            user_A = np.load(user_A_file)
            print('{:*^40}'.format('user_A'))
            print(np.sum(user_A == 1))
            return user_A
        
        adj = self.get_R_mat()
        user_num = adj.shape[0]
        user_A = np.zeros((user_num,user_num))
        # print(user_num)
        for node in range(user_num):
            user_neighbors = set()
            item_neighbors = np.nonzero(adj[[node],:])[-1].tolist()  # 获取邻居节点
            print("第{}为用户的项目邻居是{}".format(node, item_neighbors))
            for neighbor in item_neighbors:
                users = np.nonzero(adj[:,[neighbor]])[0].tolist()
                if len(users) == 0:
                    continue
                user_neighbors = user_neighbors.union(users)
            user_A[node,np.array(list(user_neighbors)).astype(int)] = 1

        np.save(user_A_file, user_A)
        return user_A
    
    def get_zhipu_user_A(self):
        A = np.matmul(self.user_embedding, self.user_embedding.T)
        A_min, A_max = np.min(A), np.max(A)
        A_normalized = (A - A_min) / (A_max - A_min)
        user_A =  np.zeros_like(A)
        user_A[A_normalized > self.threshold] = 1
        print('{:*^40}'.format('zhipu_user_A'))
        print(np.sum(A_normalized > self.threshold))
        print(np.sum(user_A == 1))
        return user_A
    
    def get_zhipu_item_A(self):
        threshold = 0.6
        A = np.matmul(self.item_embedding, self.item_embedding.T)
        A_min, A_max = np.min(A), np.max(A)
        A_normalized = (A - A_min) / (A_max - A_min)
        item_A =  np.zeros_like(A)
        item_A[A_normalized > self.threshold] = 1
        print('{:*^40}'.format('zhipu_item_A'))
        print(np.sum(A_normalized > self.threshold))
        print(np.sum(item_A == 1))
        return item_A
    
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        # return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
        return scipy_sparse_mat_to_torch_sparse_tensor(adj_mat.tocsr()), scipy_sparse_mat_to_torch_sparse_tensor(norm_adj_mat.tocsr()), scipy_sparse_mat_to_torch_sparse_tensor(mean_adj_mat.tocsr())

    def read_neg_file(self,path):
        try:
            test_neg = path + '/test_neg.txt'
            test_neg_f = open(test_neg ,'r')
        except:
            negativeList = None
            return negativeList
        negativeList = []
        line = test_neg_f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:  # arr[0] = (user, pos_item)
                item = int(x)
                negatives.append(item)
            negativeList.append(negatives)
            line = test_neg_f.readline()
        return negativeList
    def get_test_neg_item(self,u,negativeList):
        neg_items = negativeList[u]
        return neg_items
    def get_train_instance(self):
        user_input, item_input, labels = [],[],[]
        for (u, i) in self.R.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative negRatio instances
            for _ in range(self.neg_num):
                j = np.random.randint(self.n_items)
                while (u, j) in self.R.keys():
                    j = np.random.randint(self.n_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return np.array(user_input),np.array(item_input),np.array(labels)

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self,save_log):
        pprint('n_users=%d, n_items=%d' % (self.n_users, self.n_items),save_log)
        pprint('n_interactions=%d' % (self.n_train + self.n_test),save_log)
        pprint('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)),save_log)


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split('\t')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write('\t'.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state


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
