import math
import heapq 
import multiprocessing
import numpy as np
from time import time
from tqdm import tqdm
from collections import defaultdict
import torch

_model = None
_testRatings = None
_testNegatives = None
_sess = None
_data_type = None
_layer_size = None
_test_user_list = None

def test(model, data_generator, test_user_list, data_type,batch_size,ks, layer_size):
    global _model
    global _sess
    global _testRatings
    global _testNegatives
    global _layer_size
    global _data_type
    global _test_user_list

    _model = model
    _testRatings = np.array(data_generator.ratingList)
    _testNegatives = data_generator.negativeList
    _layer_size = layer_size
    _data_type = data_type
    _test_user_list = test_user_list

    hits, ndcgs = [],[]

    users,items,user_gt_item = get_test_instance()
    num_test_batches = len(users)//batch_size
    # bar_test = ProgressBar('test_'+_data_type, max=num_test_batches+1)
    # sample_id = 0
    test_preds = []
    if _data_type == 'source':
        for current_batch in tqdm(range(num_test_batches+1),desc='test_source',ascii=True):
            # bar_test.next()
            min_idx = current_batch*batch_size
            max_idx = np.min([(current_batch+1)*batch_size,len(users)])
            batch_input_users = users[min_idx:max_idx]
            batch_input_items = items[min_idx:max_idx]
            with torch.no_grad():
                predictions = model.predict(batch_input_users, batch_input_items, "source")
                test_preds.extend(predictions.cpu())
        assert len(test_preds)==len(users),'source num is not equal'
    else:
        for current_batch in tqdm(range(num_test_batches+1),desc='test_target',ascii=True):
            # bar_test.next()
            min_idx = current_batch*batch_size
            max_idx = np.min([(current_batch+1)*batch_size,len(users)])
            batch_input_users = users[min_idx:max_idx]
            batch_input_items = items[min_idx:max_idx]
            # print(max(batch_input_items))
            # print(min(batch_input_items))
            # print(max(batch_input_users))
            # print(min(batch_input_users))
            with torch.no_grad():
                predictions = model.predict(batch_input_users, batch_input_items,"target")
                test_preds.extend(predictions.cpu())
        assert len(test_preds)==len(users),'target num is not equal'

    user_item_preds = defaultdict(lambda: defaultdict(float))
    user_pred_gtItem = defaultdict(float)
    for sample_id in range(len(users)):
        user = users[sample_id]
        item = items[sample_id]
        pred = test_preds[sample_id]  # [pos_prob, neg_prob]
        user_item_preds[user][item] = pred
    for user in user_item_preds.keys():
        item_pred = user_item_preds[user]
        hrs,nds=[],[]
        for k in ks:
            ranklist = heapq.nlargest(k, item_pred, key=item_pred.get)
            hr = getHitRatio(ranklist, user_gt_item[user])
            ndcg = getNDCG(ranklist, user_gt_item[user])
            hrs.append(hr)
            nds.append(ndcg)
        hits.append(hrs)
        ndcgs.append(nds)
    hr, ndcg = np.array(hits).mean(axis=0), np.array(ndcgs).mean(axis=0)
    return (hr, ndcg)
        
def get_test_instance():
    users,items = [],[]
    user_gt_item = {}
    for idx in _test_user_list:
        rating = _testRatings[idx]
        items_neg = _testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        user_gt_item[u] = gtItem
        for item in items_neg:
            users.append(u)
            items.append(item)
        items.append(gtItem)
        users.append(u)
    return np.array(users),np.array(items),user_gt_item

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

