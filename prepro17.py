import itertools
from collections import OrderedDict
import json
from typing import List
import time


from fsspec import transaction
from torch.utils import data
from collections import defaultdict
from tqdm import tqdm
#from transformers.models.auto.configuration_auto import F
import ujson as json
import os
import pickle
import random
import numpy as np
import gc
docred_rel2id = json.load(open('./meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

#import numba as nb
#from transformers import AutoConfig, AutoModel, AutoTokenizer
from itertools import chain
import joblib

import gc
def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def extract_path(data, keep_sent_order):
    sents = data["sents"]#sents是一维列表，["str1","str2",...,"strN"]
    nodes = [[] for _ in range(len(data['sents']))]#[[],[],[],..(总共N个)..,[]]
    e2e_sent = defaultdict(dict)#{}

    # create mention's list for each sentence 给每个句子创建一个mention-list
    for ns_no, ns in enumerate(data['vertexSet']):#ns_no:实体的编号，ns：一个实体 [{mention}{}...{}]

        for n in ns:#n:{'name':str,'sent_id':num,'pos':(s,e),'type':str}
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)
            #[
            #   [1,1,4,6,8]
            #   [1,4,6,7]
            #   [2,6]
            #   ...
            #   [1,2,5,6,8]
            # ]
            # 上面的数字代表实体编号

    for sent_id in range(len(sents)):#sents是一维列表，["str1","str2",...,"strN"]
        for n1 in nodes[sent_id]:#nodes[sent_id]:"1,1,4,6,8"，n1:1or4or...
            for n2 in nodes[sent_id]:#n2
                if n1 == n2:#不与自身比较,stri==strj"
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()#set函数 的意义？
                e2e_sent[n1][n2].add(sent_id)#{(1,2):8,(1,4):{1,2},}表示联系起1,2号实体的有8号句子；联系起1,4号实体的有1,2号句子5

    # 2-hop Path
    path_two = defaultdict(dict)
    #path_two:{}
    entityNum = len(data['vertexSet'])#实体的数量
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue
                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        if s1 == s2:
                            continue
                        if n2 not in path_two[n1]:
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        if keep_sent_order == True:
                            cand_sents.sort()
                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop Path
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    for cand1 in e2e_sent[n1][n3]:
                        for cand2 in path_two[n3][n2]:
                            if cand1 in cand2[0]:
                                continue
                            if cand2[1] == n1:
                                continue
                            if n2 not in path_three[n1]:
                                path_three[n1][n2] = []
                            cand_sents = [cand1] + cand2[0]
                            if keep_sent_order:
                                cand_sents.sort()
                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive Path
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 2:
                        continue
                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    # Merge 匹配、拼接
    merge = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n2 in path_two[n1]:
                merge[n1][n2] = path_two[n1][n2]
            if n2 in path_three[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += path_three[n1][n2]
                else:
                    merge[n1][n2] = path_three[n1][n2]

            if n2 in consecutive[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += consecutive[n1][n2]
                else:
                    merge[n1][n2] = consecutive[n1][n2]

    # Default Path
    for h in range(len(data['vertexSet'])):
        for t in range(len(data['vertexSet'])):
            if h == t:
                continue
            if t in merge[h]:
                continue
            merge[h][t] = []
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    cand_sents = [n1['sent_id'], n2['sent_id']]
                    if keep_sent_order:
                        cand_sents.sort()
                    merge[h][t].append([cand_sents])

    # Remove redundency
    tp_set = set()
    for n1 in merge.keys():
        for n2 in merge[n1].keys():
            hash_set = set()
            new_list = []
            for t in merge[n1][n2]:
                if tuple(t[0]) not in hash_set:
                    hash_set.add(tuple(t[0]))
                    new_list.append(t[0])
            merge[n1][n2] = new_list


    #得到路径集合和头尾实体对集合
    path2en = defaultdict(dict)
    e2e_list=[]
    path_list=[]
    path_set_list=[]
    for n1 in range(entityNum):
        for n2 in range(n1+1,entityNum):
            e2e_list.append([n1,n2].__str__())
            # print(merge[n1][n2])
            for path in merge[n1][n2]:
                path_list.append(path)



    #print("path_list:",path_list)
    # 展平列表
    flattened_list = [item for sublist in path_list for item in sublist]

    # 去重并排序
    unique_sorted_list = sorted(set(flattened_list))




    return merge,nodes,unique_sorted_list

#输入：req_skills = ["java","nodejs","reactjs"], people = [["java"],["nodejs"],["nodejs","reactjs"]]
#现在有e2e_list: ['[0, 1]', '[0, 2]', '[0, 3]', '[0, 4]', '[0, 5]', '[0, 6]', '[0, 7]', '[0, 8]', '[0, 9]', '[0, 10]', '[0, 11]', '[0, 12]', '[0, 13]', '[0, 14]', '[0, 15]', '[0, 16]', '[1, 2]', '[1, 3]', '[1, 4]', '[1, 5]', '[1, 6]', '[1, 7]', '[1, 8]', '[1, 9]', '[1, 10]', '[1, 11]', '[1, 12]', '[1, 13]', '[1, 14]', '[1, 15]', '[1, 16]', '[2, 3]', '[2, 4]', '[2, 5]', '[2, 6]', '[2, 7]', '[2, 8]', '[2, 9]', '[2, 10]', '[2, 11]', '[2, 12]', '[2, 13]', '[2, 14]', '[2, 15]', '[2, 16]', '[3, 4]', '[3, 5]', '[3, 6]', '[3, 7]', '[3, 8]', '[3, 9]', '[3, 10]', '[3, 11]', '[3, 12]', '[3, 13]', '[3, 14]', '[3, 15]', '[3, 16]', '[4, 5]', '[4, 6]', '[4, 7]', '[4, 8]', '[4, 9]', '[4, 10]', '[4, 11]', '[4, 12]', '[4, 13]', '[4, 14]', '[4, 15]', '[4, 16]', '[5, 6]', '[5, 7]', '[5, 8]', '[5, 9]', '[5, 10]', '[5, 11]', '[5, 12]', '[5, 13]', '[5, 14]', '[5, 15]', '[5, 16]', '[6, 7]', '[6, 8]', '[6, 9]', '[6, 10]', '[6, 11]', '[6, 12]', '[6, 13]', '[6, 14]', '[6, 15]', '[6, 16]', '[7, 8]', '[7, 9]', '[7, 10]', '[7, 11]', '[7, 12]', '[7, 13]', '[7, 14]', '[7, 15]', '[7, 16]', '[8, 9]', '[8, 10]', '[8, 11]', '[8, 12]', '[8, 13]', '[8, 14]', '[8, 15]', '[8, 16]', '[9, 10]', '[9, 11]', '[9, 12]', '[9, 13]', '[9, 14]', '[9, 15]', '[9, 16]', '[10, 11]', '[10, 12]', '[10, 13]', '[10, 14]', '[10, 15]', '[10, 16]', '[11, 12]', '[11, 13]', '[11, 14]', '[11, 15]', '[11, 16]', '[12, 13]', '[12, 14]', '[12, 15]', '[12, 16]', '[13, 14]', '[13, 15]', '[13, 16]', '[14, 15]', '[14, 16]', '[15, 16]']
#and Path=[['[10, 15]', '[10, 16]'], ['[1, 6]', '[1, 7]'], ['[0, 1]', '[0, 2]', '[0, 3]', '[0, 13]', '[0, 14]', '[1, 4]', '[1, 8]', '[1, 11]', '[1, 13]', '[1, 14]', '[2, 4]', '[2, 8]', '[2, 11]', '[2, 13]', '[2, 14]', '[3, 4]', '[3, 8]', '[3, 11]', '[3, 13]', '[3, 14]', '[4, 8]', '[4, 11]', '[8, 12]', '[8, 13]', '[8, 14]', '[11, 12]', '[11, 13]', '[11, 14]', '[12, 13]', '[12, 14]'], ['[0, 4]', '[0, 8]', '[0, 11]', '[0, 12]', '[4, 12]', '[4, 13]', '[4, 14]', '[8, 12]', '[8, 13]', '[8, 14]', '[11, 12]', '[11, 13]', '[11, 14]', '[12, 13]', '[12, 14]'], ['[0, 8]', '[0, 9]', '[1, 8]', '[1, 9]', '[2, 8]', '[2, 9]', '[3, 8]', '[3, 9]', '[4, 8]', '[4, 9]'], ['[1, 15]', '[1, 16]', '[2, 15]', '[2, 16]', '[3, 15]', '[3, 16]'], ['[0, 6]', '[0, 7]', '[2, 6]', '[2, 7]', '[3, 6]', '[3, 7]', '[4, 6]', '[4, 7]'], ['[6, 12]', '[6, 13]', '[6, 14]', '[7, 12]', '[7, 13]', '[7, 14]'], ['[1, 15]', '[1, 16]', '[5, 15]', '[5, 16]'], ['[0, 5]', '[2, 5]', '[3, 5]', '[4, 5]'], ['[4, 9]', '[9, 11]', '[9, 12]'], ['[5, 8]', '[5, 9]', '[6, 8]', '[6, 9]', '[7, 8]', '[7, 9]'], ['[4, 15]', '[4, 16]', '[8, 15]', '[8, 16]', '[11, 15]', '[11, 16]', '[12, 15]', '[12, 16]'], ['[0, 15]', '[0, 16]', '[12, 15]', '[12, 16]', '[13, 15]', '[13, 16]', '[14, 15]', '[14, 16]'], ['[5, 6]', '[5, 7]', '[6, 7]'], ['[1, 5]'], ['[6, 11]', '[6, 12]', '[7, 11]', '[7, 12]'], ['[4, 8]', '[4, 11]', '[4, 12]', '[8, 11]', '[8, 12]', '[11, 12]'], ['[0, 9]', '[1, 9]', '[2, 9]', '[3, 9]'], ['[0, 10]', '[1, 10]', '[2, 10]', '[3, 10]'], ['[9, 10]'], ['[15, 16]'], ['[4, 8]', '[4, 10]', '[4, 11]', '[4, 12]', '[8, 10]', '[10, 11]', '[10, 12]'], ['[6, 10]', '[7, 10]'], ['[1, 5]', '[1, 6]', '[1, 7]', '[5, 6]', '[5, 7]'], ['[0, 8]', '[0, 11]', '[0, 12]', '[1, 8]', '[1, 11]', '[1, 12]', '[2, 8]', '[2, 11]', '[2, 12]', '[3, 8]', '[3, 11]', '[3, 12]'], ['[1, 8]', '[1, 9]', '[5, 8]', '[5, 9]'], ['[0, 9]', '[9, 13]', '[9, 14]'], ['[5, 12]', '[5, 13]', '[5, 14]'], ['[0, 4]', '[0, 10]', '[4, 12]', '[4, 13]', '[4, 14]', '[10, 12]', '[10, 13]', '[10, 14]'], ['[0, 5]', '[0, 6]', '[0, 7]', '[1, 5]', '[1, 6]', '[1, 7]', '[2, 5]', '[2, 6]', '[2, 7]', '[3, 5]', '[3, 6]', '[3, 7]', '[4, 5]', '[4, 6]', '[4, 7]'], ['[9, 15]', '[9, 16]'], ['[0, 1]', '[0, 2]', '[0, 3]', '[0, 4]', '[1, 2]', '[1, 3]', '[1, 4]', '[2, 3]', '[2, 4]', '[3, 4]'], ['[5, 10]'], ['[8, 9]'], ['[5, 8]', '[5, 11]', '[5, 12]'], ['[4, 10]'], ['[1, 12]', '[1, 13]', '[1, 14]', '[2, 12]', '[2, 13]', '[2, 14]', '[3, 12]', '[3, 13]', '[3, 14]', '[4, 12]', '[4, 13]', '[4, 14]'], ['[0, 12]', '[0, 13]', '[0, 14]', '[12, 13]', '[12, 14]', '[13, 14]'], ['[5, 15]', '[5, 16]', '[6, 15]', '[6, 16]', '[7, 15]', '[7, 16]'], ['[1, 4]', '[1, 8]', '[1, 11]', '[1, 12]', '[4, 5]', '[5, 8]', '[5, 11]', '[5, 12]'], ['[4, 8]', '[4, 9]', '[8, 10]', '[9, 10]'], ['[1, 4]', '[1, 10]', '[4, 5]', '[5, 10]'], ['[10, 12]', '[10, 13]', '[10, 14]']]
#输出：[0,2]
def smallestSufficientTeam_fast(req_skills: List[str], people: List[List[str]]) -> List[int]:
    # 初始化
    time_start_func=time.time()
    req_skills_set = set(req_skills)
    skills_dict = {x: [] for x in req_skills}
    for i, p in enumerate(people):
        for skill in p:
            if skill in req_skills_set:
                skills_dict[skill].append(i)
    skills_set_dict = {k: set(v) for k, v in skills_dict.items()}
    # 排序，更早搜索到最终解
    for k, v in skills_dict.items():
        v.sort(key=lambda i: len(people[i]))
    req_skills.sort(key=lambda x: len(skills_dict[x]))


    # dfs
    res = len(req_skills)
    res_people = [x[0] for x in skills_dict.values()]
    now_people = []
    s = [[0, skills_dict[req_skills[0]]]]
    while len(s) > 0:
        time_end_func=time.time()
        time_cost_func=time_end_func-time_start_func
        if time_cost_func>60:
            return res_people

        if len(s[-1][-1]) > 0:
            p = s[-1][0] + 1
            person = s[-1][-1].pop()
            # 剪枝
            if len(now_people) >= res - 1:
                s.pop()
                if now_people:
                    now_people.pop()
                continue
            now_people.append(person)
            now_people_set = set(now_people)
            while p < len(req_skills) and len(skills_set_dict[req_skills[p]] & now_people_set) > 0:
                p += 1
            if p == len(req_skills):
                if len(now_people) <= res:
                    res = len(now_people)
                    res_people = now_people[:]
                if now_people:
                    now_people.pop()
            else:
                s.append([p, skills_dict[req_skills[p]][:]])
        else:
            s.pop()
            if len(now_people) == 0:
                break
            if now_people:
                now_people.pop()
    return res_people
class ReadDataset:
    def __init__(self, dataset: str, tokenizer, max_seq_Length: int = 1024,
             transformers: str = 'bert') -> None:
        self.transformers = transformers
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_Length = max_seq_Length

    def read(self, file_in: str):
        save_file = file_in.split('.json')[0] + '_' + self.transformers + '_' \
                        + self.dataset + '.pkl'
        if self.dataset == 'docred':
            return read_docred(self.transformers, file_in, save_file, self.tokenizer, self.max_seq_Length)
        elif self.dataset == 'cdr':
            return read_cdr(file_in, save_file, self.tokenizer, self.max_seq_Length)
        elif self.dataset == 'gda':
            return read_gda(file_in, save_file, self.tokenizer, self.max_seq_Length)
        else:
            raise RuntimeError("No read func for this dataset.")



def read_docred(transfermers, file_in, save_file, tokenizer, max_seq_length=1024):
    timeinit=time.time()
    if os.path.exists(save_file):
        with open(file=save_file, mode='rb') as fr:
            features = joblib.load(fr)
            fr.close()
        # with tqdm(total=os.path.getsize(save_file)) as pbar:
        #     with open(file=save_file, mode='rb') as fr:
        #         features = joblib.load(fr)
        #         for l in fr:
        #             pbar.updata(len(l))
        #         fr.close()

        print('load preprocessed data from {}.'.format(save_file))
        return features
    else:
        features = []
        pos_samples = 0
        neg_samples = 0

        if file_in == "":
            return None
        with open(file_in, "r") as fh:
            data = json.load(fh)
        fh.close()

        if transfermers == 'bert':
            # entity_type = ["ORG", "-",  "LOC", "-",  "TIME", "-",  "PER", "-", "MISC", "-", "NUM"]
            entity_type = ["-", "ORG", "-", "LOC", "-", "TIME", "-", "PER", "-", "MISC", "-", "NUM"]

        sumfuture = 0
        # for sample in tqdm(data, desc="Example"):
        #for i, sample in enumerate(data):
        for sample in tqdm(data, desc="Example"):


            merge, nodes, min_path_list = extract_path(sample, True)

            entities = sample['vertexSet']
            sentss = sample["sents"]

            ids = list(map(nodes.__getitem__, min_path_list))
            select_entity_idxs = list(chain.from_iterable(ids))
            select_entity_idx = list(set(select_entity_idxs))
            select_entity_idx.sort()

            # 根据实体id 找实体
            select_entity = list(map(entities.__getitem__, select_entity_idx))
            select_sentence = list(map(sentss.__getitem__, min_path_list))

            entity_start, entity_end = [], []
            mention_types = []
            # for entity in entities:
            for entity in select_entity:
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0]))
                    entity_end.append((sent_id, pos[1] - 1))
                    mention_types.append(mention['type'])

            sents = []

            sent_map = []

            for i_s, sent in enumerate(sample['sents']):
                new_map = {}
                for i_t, token in enumerate(sent):
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if (i_s, i_t) in entity_start:
                        t = entity_start.index((i_s, i_t))
                        if transfermers == 'bert':
                            mention_type = mention_types[t]
                            special_token_i = entity_type.index(mention_type)
                            special_token = ['[unused' + str(special_token_i) + ']']
                        else:
                            special_token = ['*']
                        tokens_wordpiece = special_token + tokens_wordpiece
                        # tokens_wordpiece = ["[unused0]"]+ tokens_wordpiece

                    if (i_s, i_t) in entity_end:
                        t = entity_end.index((i_s, i_t))
                        if transfermers == 'bert':
                            mention_type = mention_types[t]
                            special_token_i = entity_type.index(mention_type) + 50
                            special_token = ['[unused' + str(special_token_i) + ']']
                        else:
                            special_token = ['*']
                        tokens_wordpiece = tokens_wordpiece + special_token

                        # tokens_wordpiece = tokens_wordpiece + ["[unused1]"]
                        # print(tokens_wordpiece,tokenizer.convert_tokens_to_ids(tokens_wordpiece))

                    new_map[i_t] = len(sents)
                    sents.extend(tokens_wordpiece)
                new_map[i_t + 1] = len(sents)
                sent_map.append(new_map)

            # sents_bufen=[]
            # for i_s, sent in enumerate(select_sentence):
            #     for i_t, token in enumerate(sent):
            #         tokens_wordpiece = tokenizer.tokenize(token)
            #         if (i_s, i_t) in entity_start:
            #             t = entity_start.index((i_s, i_t))
            #             if transfermers == 'bert':
            #                 mention_type = mention_types[t]
            #                 special_token_i = entity_type.index(mention_type)
            #                 special_token = ['[unused' + str(special_token_i) + ']']
            #             else:
            #                 special_token = ['*']
            #             tokens_wordpiece = special_token + tokens_wordpiece
            #             # tokens_wordpiece = ["[unused0]"]+ tokens_wordpiece
            #
            #         if (i_s, i_t) in entity_end:
            #             t = entity_end.index((i_s, i_t))
            #             if transfermers == 'bert':
            #                 mention_type = mention_types[t]
            #                 special_token_i = entity_type.index(mention_type) + 50
            #                 special_token = ['[unused' + str(special_token_i) + ']']
            #             else:
            #                 special_token = ['*']
            #             tokens_wordpiece = tokens_wordpiece + special_token
            #
            #             # tokens_wordpiece = tokens_wordpiece + ["[unused1]"]
            #             # print(tokens_wordpiece,tokenizer.convert_tokens_to_ids(tokens_wordpiece))
            #
            #
            #         sents_bufen.extend(tokens_wordpiece)

            entity_pos = []
            # for e in entities:
            for e in entities:
                entity_pos.append([])
                mention_num = len(e)
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))

                    # 选择句子中所有label
            labels_select = []
            if "labels" in sample:
                for label in sample['labels']:
                    if label['h'] in select_entity_idx and label['t'] in select_entity_idx:
                        labels_select.append(label)
                # if len(labels_select) == 0:
                #     continue

            train_triple = {}
            relations, hts = [], []
            if "labels" in sample:
                for label in labels_select:
                    # for label in sample['labels']:
                    evidence = label['evidence']
                    r = int(docred_rel2id[label['r']])
                    if (label['h'], label['t']) not in train_triple:
                        train_triple[(label['h'], label['t'])] = [
                            {'relation': r, 'evidence': evidence}]
                    else:
                        train_triple[(label['h'], label['t'])].append(
                            {'relation': r, 'evidence': evidence})
            ##if "labels" in sample:
            # if "labels" in sample:

            # Get positive samples from dataset
            for h, t in train_triple.keys():
                relation = [0] * len(docred_rel2id)
                for mention in train_triple[h, t]:
                    relation[mention["relation"]] = 1
                    # evidence = mention["evidence"]
                relations.append(relation)
                hts.append([h, t])
                pos_samples += 1

            # Get negative samples from dataset
            for h in range(len(select_entity)):
                for t in range(len(select_entity)):
                    if h != t and [h, t] not in hts:
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        relations.append(relation)
                        hts.append([h, t])
                        neg_samples += 1

            # else:  # 无标签实体对的处理方式与有标签实体对相同
            #     for h in range(len(entities)):
            #         for t in range(len(entities)):
            #             if h != t:
            #                # relation = [1] + [0] * (len(docred_rel2id) - 1)
            #                 relation = [1] * (len(docred_rel2id) - 1)
            #                 relations.append(relation)
            #                 hts.append([h, t])
            #                 neg_samples += 1

            # assert len(relations) == len(entities) * (len(entities) - 1)

            if len(hts) == 0:
                print(len(sent))
            sents = sents[:max_seq_length - 2]
            # sents = sents_bufen[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            feature = {'input_ids': input_ids,
                       'entity_pos': entity_pos,
                       'labels': relations,
                       'hts': hts,
                       'title': sample['title'],
                       }
            features.append(feature)

            # else:
            #     default_relation = [1]+[0] * (len(cdr_rel2id)-1)
            #     feature = {'input_ids': input_ids,
            #                'entity_pos': entity_pos,
            #                'labels': [default_relation],
            #                'hts': [[-1, -1]],  # 默认的索引值为-1
            #                'title': sample['title'],
            #                }
            #     features.append(feature)

        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
        # print("# {} examples len>512 and max len is {}.".format(up512_num, max_len))

        end=time.time()
        totaldatasettime=end-timeinit
        print('total-dataset-time=.'.format(totaldatasettime))
        with open(file=save_file, mode='wb') as fw:
            pickle.dump(features, fw)
        print('finish reading {} and save preprocessed data to {}.'.format(file_in, save_file))

        return features





def read_cdr(file_in, save_file, tokenizer, max_seq_length=1024):
    if os.path.exists(save_file):
        with open(file=save_file, mode='rb') as fr:
            features = pickle.load(fr)
            fr.close()
        print('load preprocessed data from {}.'.format(save_file))
        return features
    else:
        pmids = set()
        features = []
        maxlen = 0
        with open(file_in, 'r') as infile:
            lines = infile.readlines()
            for i_l, line in enumerate(tqdm(lines)):
                line = line.rstrip().split('\t')
                pmid = line[0]

                if pmid not in pmids:
                    pmids.add(pmid)
                    text = line[1]
                    prs = chunks(line[2:], 17)

                    ent2idx = {}
                    train_triples = {}

                    entity_pos = set()
                    for p in prs:
                        es = list(map(int, p[8].split(':')))
                        ed = list(map(int, p[9].split(':')))
                        tpy = p[7]
                        for start, end in zip(es, ed):
                            entity_pos.add((start, end, tpy))

                        es = list(map(int, p[14].split(':')))
                        ed = list(map(int, p[15].split(':')))
                        tpy = p[13]
                        for start, end in zip(es, ed):
                            entity_pos.add((start, end, tpy))

                    sents = [t.split(' ') for t in text.split('|')]
                    new_sents = []
                    sent_map = {}
                    i_t = 0
                    for sent in sents:
                        for token in sent:
                            tokens_wordpiece = tokenizer.tokenize(token)
                            for start, end, tpy in list(entity_pos):
                                if i_t == start:
                                    tokens_wordpiece = ["*"] + tokens_wordpiece
                                if i_t + 1 == end:
                                    tokens_wordpiece = tokens_wordpiece + ["*"]
                            sent_map[i_t] = len(new_sents)
                            new_sents.extend(tokens_wordpiece)
                            i_t += 1
                        sent_map[i_t] = len(new_sents)
                    sents = new_sents

                    entity_pos = []

                    for p in prs:
                        if p[0] == "not_include":
                            continue
                        if p[1] == "L2R":
                            h_id, t_id = p[5], p[11]
                            h_start, t_start = p[8], p[14]
                            h_end, t_end = p[9], p[15]
                        else:
                            t_id, h_id = p[5], p[11]
                            t_start, h_start = p[8], p[14]
                            t_end, h_end = p[9], p[15]
                        h_start = map(int, h_start.split(':'))
                        h_end = map(int, h_end.split(':'))
                        t_start = map(int, t_start.split(':'))
                        t_end = map(int, t_end.split(':'))
                        h_start = [sent_map[idx] for idx in h_start]
                        h_end = [sent_map[idx] for idx in h_end]
                        t_start = [sent_map[idx] for idx in t_start]
                        t_end = [sent_map[idx] for idx in t_end]
                        if h_id not in ent2idx:
                            ent2idx[h_id] = len(ent2idx)
                            entity_pos.append(list(zip(h_start, h_end)))
                        if t_id not in ent2idx:
                            ent2idx[t_id] = len(ent2idx)
                            entity_pos.append(list(zip(t_start, t_end)))
                        h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                        r = cdr_rel2id[p[0]]
                        if (h_id, t_id) not in train_triples:
                            train_triples[(h_id, t_id)] = [{'relation': r}]
                        else:
                            train_triples[(h_id, t_id)].append({'relation': r})

                    relations, hts = [], []
                    for h, t in train_triples.keys():
                        relation = [0] * len(cdr_rel2id)
                        for mention in train_triples[h, t]:
                            relation[mention["relation"]] = 1
                        relations.append(relation)
                        hts.append([h, t])

                maxlen = max(maxlen, len(sents))
                sents = sents[:max_seq_length - 2]
                input_ids = tokenizer.convert_tokens_to_ids(sents)
                input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

                if len(hts) > 0:
                    feature = {'input_ids': input_ids,
                               'entity_pos': entity_pos,
                               'labels': relations,
                               'hts': hts,
                               'title': pmid,
                               }
                    features.append(feature)
        print("Number of documents: {}.".format(len(features)))
        print("Max document length: {}.".format(maxlen))

        with open(file=save_file, mode='wb') as fw:
            pickle.dump(features, fw)
        print('finish reading {} and save preprocessed data to {}.'.format(file_in, save_file))

        return features


def read_gda(file_in, save_file, tokenizer, max_seq_length=1024):
    if os.path.exists(save_file):
        with open(file=save_file, mode='rb') as fr:
            features = pickle.load(fr)
            fr.close()
        print('load preprocessed data from {}.'.format(save_file))
        return features
    else:
        pmids = set()
        features = []
        maxlen = 0
        with open(file_in, 'r') as infile:
            lines = infile.readlines()
            for i_l, line in enumerate(tqdm(lines)):
                line = line.rstrip().split('\t')
                pmid = line[0]

                if pmid not in pmids:
                    pmids.add(pmid)
                    text = line[1]
                    prs = chunks(line[2:], 17)

                    ent2idx = {}
                    train_triples = {}

                    entity_pos = set()
                    for p in prs:
                        es = list(map(int, p[8].split(':')))
                        ed = list(map(int, p[9].split(':')))
                        tpy = p[7]
                        for start, end in zip(es, ed):
                            entity_pos.add((start, end, tpy))

                        es = list(map(int, p[14].split(':')))
                        ed = list(map(int, p[15].split(':')))
                        tpy = p[13]
                        for start, end in zip(es, ed):
                            entity_pos.add((start, end, tpy))

                    sents = [t.split(' ') for t in text.split('|')]
                    new_sents = []
                    sent_map = {}
                    i_t = 0
                    for sent in sents:
                        for token in sent:
                            tokens_wordpiece = tokenizer.tokenize(token)
                            for start, end, tpy in list(entity_pos):
                                if i_t == start:
                                    tokens_wordpiece = ["*"] + tokens_wordpiece
                                if i_t + 1 == end:
                                    tokens_wordpiece = tokens_wordpiece + ["*"]
                            sent_map[i_t] = len(new_sents)
                            new_sents.extend(tokens_wordpiece)
                            i_t += 1
                        sent_map[i_t] = len(new_sents)
                    sents = new_sents

                    entity_pos = []

                    for p in prs:
                        if p[0] == "not_include":
                            continue
                        if p[1] == "L2R":
                            h_id, t_id = p[5], p[11]
                            h_start, t_start = p[8], p[14]
                            h_end, t_end = p[9], p[15]
                        else:
                            t_id, h_id = p[5], p[11]
                            t_start, h_start = p[8], p[14]
                            t_end, h_end = p[9], p[15]
                        h_start = map(int, h_start.split(':'))
                        h_end = map(int, h_end.split(':'))
                        t_start = map(int, t_start.split(':'))
                        t_end = map(int, t_end.split(':'))
                        h_start = [sent_map[idx] for idx in h_start]
                        h_end = [sent_map[idx] for idx in h_end]
                        t_start = [sent_map[idx] for idx in t_start]
                        t_end = [sent_map[idx] for idx in t_end]
                        if h_id not in ent2idx:
                            ent2idx[h_id] = len(ent2idx)
                            entity_pos.append(list(zip(h_start, h_end)))
                        if t_id not in ent2idx:
                            ent2idx[t_id] = len(ent2idx)
                            entity_pos.append(list(zip(t_start, t_end)))
                        h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                        r = gda_rel2id[p[0]]
                        if (h_id, t_id) not in train_triples:
                            train_triples[(h_id, t_id)] = [{'relation': r}]
                        else:
                            train_triples[(h_id, t_id)].append({'relation': r})

                    relations, hts = [], []
                    for h, t in train_triples.keys():
                        relation = [0] * len(gda_rel2id)
                        for mention in train_triples[h, t]:
                            relation[mention["relation"]] = 1
                        relations.append(relation)
                        hts.append([h, t])

                maxlen = max(maxlen, len(sents))
                sents = sents[:max_seq_length - 2]
                input_ids = tokenizer.convert_tokens_to_ids(sents)
                input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

                if len(hts) > 0:
                    feature = {'input_ids': input_ids,
                               'entity_pos': entity_pos,
                               'labels': relations,
                               'hts': hts,
                               'title': pmid,
                               }
                    features.append(feature)
        print("Number of documents: {}.".format(len(features)))
        print("Max document length: {}.".format(maxlen))
        with open(file=save_file, mode='wb') as fw:
            pickle.dump(features, fw)
        print('finish reading {} and save preprocessed data to {}.'.format(file_in, save_file))

        return features
