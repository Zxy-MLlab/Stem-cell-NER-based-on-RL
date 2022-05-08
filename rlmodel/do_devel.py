import re
import json
import numpy as np

def read_data(read_path):
    sentence_label_list = []
    with open(read_path, 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            sentence_label_json = {
                'sentence': '',
                'label': '',
            }
            if not data_line:
                break
            data_json = json.loads(data_line)
            sentence_label_json['sentence'] = data_json['sentence']
            try:
                sentence_label_json['label'] = data_json['label']
            except:
                sentence_label_json['label'] = data_json['word']
            sentence_label_list.append(sentence_label_json)

    return sentence_label_list

def read_select_data(x):
    if x == 'precise':
        select_data_path = 'output(shiyan)/sentence_label(precise).json'
    if x == 'train':
        select_data_path = 'output(shiyan)/sentence_label(train).json'
    if x == 'test':
        select_data_path = 'output(shiyan)/sentence_label(test).json'
    select_sentence_label = read_data(select_data_path)
    return select_sentence_label

def read_train_data(x):
    # train_data_path = 'data/all_train_sentence_label.json'
    # train_data = read_data(train_data_path)
    #正确标签测试集
    if x == 'precise':
        train_data_path = 'test_data/sentence_label.npy'
        train_data_numpy = np.load(train_data_path, allow_pickle=True)

    #训练数据集
    if x == 'train':
        train_data_path = 'data/all_train_sentence_label.npy'
        train_data_numpy = np.load(train_data_path, allow_pickle=True)

    #测试数据集
    if x == 'test':
        train_data_path = 'data/all_test_sentence_label.npy'
        train_data_numpy = np.load(train_data_path, allow_pickle=True)
    train_data = []
    for train_data_opt in train_data_numpy:
        for train_data_temp in train_data_opt:
            train_data.append(train_data_temp)
    return train_data

def read_true_data(x):
    #正确标签测试集
    if x == 'precise':
        true_data_path = 'test_data/train_true.json'
        true_sentence_label = read_data(true_data_path)

    #训练数据集
    if x == 'train':
        true_data_path = 'data/train_true.json'
        true_sentence_label = read_data(true_data_path)

    #测试数据集
    if x == 'test':
        true_data_path = 'data/train_true.json'
        true_sentence_label = read_data(true_data_path)

    return true_sentence_label

def get_precision(select_sentence_label, true_sentence_label):
    TP,FP = 0,0
    for sentence_label_temp in select_sentence_label:
        sentence = sentence_label_temp['sentence']
        pre_label = sentence_label_temp['label']
        true_label = []
        for true_sentence_label_temp in true_sentence_label:
            if sentence == true_sentence_label_temp['sentence']:
                true_label = true_sentence_label_temp['label']
                break
        if pre_label == true_label:
            TP = TP + 1
        else:
            FP = FP + 1
    precision = float(TP) / float(TP+FP)
    return precision

def get_recall(train_data, select_data, true_data):
    s_TP,t_TP = 0,0
    for sentence_label_temp in select_data:
        sentence = sentence_label_temp['sentence']
        pre_label = sentence_label_temp['label']
        true_label = []
        for true_sentence_label_temp in true_data:
            if sentence == true_sentence_label_temp['sentence']:
                true_label = true_sentence_label_temp['label']
                break
        if pre_label == true_label:
            s_TP = s_TP + 1
        else:
            continue

    for sentence_label_temp in train_data:
        sentence = sentence_label_temp['sentence']
        train_label = sentence_label_temp['label']
        true_label = []
        for true_sentence_label_temp in true_data:
            if sentence == true_sentence_label_temp['sentence']:
                true_label = true_sentence_label_temp['label']
                break
        if train_label == true_label:
            t_TP = t_TP+1
        else:
            continue

    recall = float(s_TP) / float(s_TP+t_TP)
    return recall

def get_attention_accurary():
    pre_data_path = 'data/sentence_label.json'
    pre_data = read_data(pre_data_path)
    true_data_path = 'data/train_true.json'
    true_data = read_data(true_data_path)
    TP,FN = 0,0
    for pre_data_temp in pre_data:
        sentence = pre_data_temp['sentence']
        pre_label = pre_data_temp['label']
        true_label = []
        for true_data_temp in true_data:
            true_label = []
            if sentence == true_data_temp['sentence']:
                true_label = true_data_temp['label']
                break
        if pre_label == true_label:
            TP = TP + 1
        else:
            FN = FN + 1
    accurary = float(TP) / float(TP+FN)
    print(accurary)
    return

def main(x):
    print("read select data!")
    select_data = read_select_data(x)
    print("read train data!")
    train_data = read_train_data(x)
    print("read true data!")
    true_data = read_true_data(x)
    print("get precision!")
    precision = get_precision(select_sentence_label=select_data,true_sentence_label=true_data)
    print("get recall!")
    recall = get_recall(train_data=train_data, select_data=select_data, true_data=true_data)
    f1 = (precision*recall*2) / (precision+recall)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    # get_attention_accurary()

    precision = precision*100
    recall = recall*100
    f1 = f1*100
    return precision, recall, f1

# if __name__ == '__main__':
#     main()