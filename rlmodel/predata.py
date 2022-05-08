import re
import json
import pickle
import numpy as np
import random

def init_label_data():
    dict_labelname2id = {}
    print("reading label data...")
    with open('data/all_sentence_label.json', 'r', encoding='utf-8') as f_read:
        while True:
            content = f_read.readline()
            if not content:
                break
            json_content = json.loads(content)
            label_name_list = json_content['label']
            for label_name_temp in label_name_list:
                label_name_temp_list = label_name_temp.split(' ')
                label_name_without_stemcell_list = label_name_temp_list[0:len(label_name_temp_list)-2]
                if len(label_name_without_stemcell_list) == 0:
                    label_name = 'NAN'
                else:
                    opt = '_'
                    label_name = opt.join(label_name_without_stemcell_list)
            if label_name not in dict_labelname2id.keys():
                dict_labelname2id[label_name] = len(dict_labelname2id)
    if 'NAN' not in dict_labelname2id:
        dict_labelname2id['NAN'] = len(dict_labelname2id)
    f_read.close()
    with open('data/dict_labelname2id.pkl', 'wb') as f_write:
        pickle.dump(dict_labelname2id, f_write)
    f_write.close()
    print("read laebl data finished!")
    return

def init_word_to_id_data():
    print('reading word embedding data...')
    vec = []
    word2id = {}
    # import the word vec
    f = open('data/vec.txt', encoding='utf-8')
    info = f.readline()
    print('word vec info:', info)
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)  # 所有词的id
        content = content[1:]
        content = [float(i) for i in content]
        vec.append(content)  # 所有词的词向量表示
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)  # 将vec装换成numpy数组

    print('reading entity to id ')
    with open('data/dict_labelname2id.pkl','rb') as input:
        dict_labelname2id = pickle.load(input)

    fixlen = 70
    maxlen = 60

    print("reading sentence label to id")
    sentence_id_list = []
    label_id_list = []
    with open('data/sentence_label.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            sentence_id_opt_list = []
            label_id_opt_list = []
            data_json = json.loads(data_line)
            sentence = data_json['sentence']
            label_list = data_json['label']
            if len(label_list) == 0:
                label = 'NAN'
            else:
                label_opt = label_list[0]
                label_opt_list = label_opt.split(' ')
                label_opt_with_stemcell_list = label_opt_list[0:len(label_opt_list)-2]
                if len(label_opt_with_stemcell_list) == 0:
                    label = 'NAN'
                else:
                    opt = '_'
                    label = opt.join(label_opt_with_stemcell_list)
            if label not in dict_labelname2id:
                label_id_opt_list = [dict_labelname2id['NAN']]
            else:
                label_id_opt_list = [dict_labelname2id[label]]

            sentence_list = sentence.split(' ')
            for i in range(min(fixlen, len(sentence_list))):
                if sentence_list[i] not in word2id:
                    sentence_id_opt_list.append(word2id['UNK'])
                else:
                    sentence_id_opt_list.append(word2id[sentence_list[i]])

            if len(sentence_id_opt_list) < fixlen:
                sentence_add_list = [word2id['BLANK'] for x in range(fixlen-len(sentence_id_opt_list))]
                sentence_id_opt_list = sentence_id_opt_list + sentence_add_list
            sentence_id_list.append(sentence_id_opt_list)
            label_id_list.append(label_id_opt_list)
    f_read.close()

    print("reading train_y")
    true_sentence_label = []
    with open('data/train_true.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            data_json = json.loads(data_line)
            sentence_label_opt_json = {
                'sentence':'',
                'label':'',
            }
            sentence_label_opt_json['sentence'] = data_json['sentence']
            sentence_label_opt_json['label'] = data_json['label']
            true_sentence_label.append(sentence_label_opt_json)
    f_read.close()

    pre_sentence_label = []
    train_y_list = []
    with open('data/sentence_label.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            data_json = json.loads(data_line)
            sentence = data_json['sentence']
            pre_label = data_json['label']
            true_label = []

            for temp in true_sentence_label:
                if sentence == temp['sentence']:
                    true_label = temp['label']
                    break
            if pre_label == true_label:
                train_y_list.append([1,0])
            else:
                train_y_list.append([0,1])
    f_read.close()

    sentence_id_numpy = np.array(sentence_id_list)
    label_id_numpy = np.array(label_id_list)
    train_y_numpy = np.array(train_y_list)

    np.save('data/all_sentence_id.npy', sentence_id_numpy)
    np.save('data/all_label_id.npy', label_id_numpy)
    np.save('data/all_train_y.npy', train_y_numpy)
    np.save('data/vec.npy', vec)

    return

def init_batch_data():
    all_sentence_label = []
    with open('data/sentence_label.json', 'r',encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            data_json = json.loads(data_line)
            all_sentence_label.append(data_json)
    all_sentence_numpy = np.load('data/all_sentence_id.npy')
    all_label_numpy = np.load('data/all_label_id.npy')
    all_train_y_numpy = np.load('data/all_train_y.npy')
    all_sentence_label_batch = []
    all_sentence_batch = []
    all_label_batch = []
    all_train_y_batch = []
    all_reward_batch = []
    sentence_label_batch = []
    sentence_batch = []
    label_batch = []
    train_y_batch = []
    reward_batch = []
    batch_num = random.randint(1,10)
    for i,sentence in enumerate(all_sentence_numpy):
        if batch_num == 0:
            all_sentence_label_batch.append(np.array(sentence_label_batch))
            all_sentence_batch.append(np.array(sentence_batch))
            all_label_batch.append(np.array(label_batch))
            all_train_y_batch.append(np.array(train_y_batch))
            all_reward_batch.append(np.array(reward_batch))
            sentence_label_batch = []
            sentence_batch = []
            label_batch = []
            train_y_batch = []
            reward_batch = []
            batch_num = random.randint(1,10)
        sentence_label_batch.append(all_sentence_label[i])
        sentence_batch.append(sentence.tolist())
        label_batch.append(all_label_numpy[i].tolist())
        train_y_batch.append(all_train_y_numpy[i].tolist())
        if all_train_y_numpy[i].tolist()[0] == 1:
            reward_batch.append(0.1)
        else:
            reward_batch.append(-0.1)
        batch_num = batch_num - 1
    all_sentence_label_batch_numpy = np.array(all_sentence_label_batch)
    all_sentence_batch_numpy = np.array(all_sentence_batch)
    all_label_batch_numpy = np.array(all_label_batch)
    all_train_y_batch_numpy = np.array(all_train_y_batch)
    all_reward_batch_numpy = np.array(all_reward_batch)
    np.save('data/sentence_label.npy', all_sentence_label_batch_numpy)
    np.save('data/sentence_id_batch.npy', all_sentence_batch_numpy)
    np.save('data/label_id_batch.npy', all_label_batch_numpy)
    np.save('data/train_y_batch.npy', all_train_y_batch_numpy)
    np.save('data/reward_batch.npy', all_reward_batch_numpy)
    return

def get_train_test_data():
    all_sentence_label = np.load('data/sentence_label.npy', allow_pickle=True)
    all_sentence_ebd = np.load('data/all_sentence_ebd.npy', allow_pickle=True)
    all_label_ebd = np.load('data/all_label_ebd.npy', allow_pickle=True)
    all_reward = np.load('data/reward_batch.npy', allow_pickle=True)
    all_train_sentence_label = []
    all_train_sentence_ebd = []
    all_train_label_ebd = []
    all_train_reward = []
    train_data_num = 3000
    for i in range(train_data_num):
        all_train_sentence_label.append(all_sentence_label[i])
        all_train_sentence_ebd.append(all_sentence_ebd[i])
        all_train_label_ebd.append(all_label_ebd[i])
        all_train_reward.append(all_reward[i])
    all_test_sentence_label = []
    all_test_sentence_ebd = []
    all_test_label_ebd = []
    all_test_reward = []
    for i in range(3100,len(all_sentence_ebd)-1):
        all_test_sentence_label.append(all_sentence_label[i])
        all_test_sentence_ebd.append(all_sentence_ebd[i])
        all_test_label_ebd.append(all_label_ebd[i])
        all_test_reward.append(all_reward[i])
    all_train_sentence_label_numpy = np.array(all_train_sentence_label)
    all_train_sentence_ebd_numpy = np.array(all_train_sentence_ebd)
    all_train_label_ebd_numpy = np.array(all_train_label_ebd)
    all_train_reward_numpy = np.array(all_train_reward)

    np.save('data/all_train_sentence_label.npy', all_train_sentence_label_numpy)
    np.save('data/all_train_sentence_ebd.npy', all_train_sentence_ebd_numpy)
    np.save('data/all_train_label_ebd.npy', all_train_label_ebd_numpy)
    np.save('data/all_train_reward.npy', all_train_reward_numpy)

    all_test_sentence_label_numpy = np.array(all_test_sentence_label)
    all_test_sentence_ebd_numpy = np.array(all_test_sentence_ebd)
    all_test_label_ebd_numpy = np.array(all_test_label_ebd)
    all_test_reward_numpy = np.array(all_test_reward)

    np.save('data/all_test_sentence_label.npy', all_test_sentence_label_numpy)
    np.save('data/all_test_sentence_ebd.npy', all_test_sentence_ebd_numpy)
    np.save('data/all_test_label_ebd.npy', all_test_label_ebd_numpy)
    np.save('data/all_test_reward.npy', all_test_reward_numpy)
    return

def get_test_data_id():
    print('reading word embedding data...')
    vec = []
    word2id = {}
    # import the word vec
    f = open('data/vec.txt', encoding='utf-8')
    info = f.readline()
    print('word vec info:', info)
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)  # 所有词的id
        content = content[1:]
        content = [float(i) for i in content]
        vec.append(content)  # 所有词的词向量表示
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    with open('data/dict_labelname2id.pkl','rb') as input:
        dict_labelname2id = pickle.load(input)

    fixlen = 70
    maxlen = 60

    print("reading sentence label to id")
    sentence_id_list = []
    label_id_list = []
    all_sentence_label = []
    with open('test_data/sentence_label.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            data_json = json.loads(data_line)
            all_sentence_label.append(data_json)
            sentence_id_opt_list = []
            label_id_opt_list = []
            sentence = data_json['sentence']
            label_list = data_json['label']
            if len(label_list) == 0:
                label = 'NAN'
            else:
                label_opt = label_list[0]
                label_opt_list = label_opt.split(' ')
                label_opt_with_stemcell_list = label_opt_list[0:len(label_opt_list) - 2]
                if len(label_opt_with_stemcell_list) == 0:
                    label = 'NAN'
                else:
                    opt = '_'
                    label = opt.join(label_opt_with_stemcell_list)
            if label not in dict_labelname2id:
                label_id_opt_list = [dict_labelname2id['NAN']]
            else:
                label_id_opt_list = [dict_labelname2id[label]]

            sentence_list = sentence.split(' ')
            for i in range(min(fixlen, len(sentence_list))):
                if sentence_list[i] not in word2id:
                    sentence_id_opt_list.append(word2id['UNK'])
                else:
                    sentence_id_opt_list.append(word2id[sentence_list[i]])

            if len(sentence_id_opt_list) < fixlen:
                sentence_add_list = [word2id['BLANK'] for x in range(fixlen - len(sentence_id_opt_list))]
                sentence_id_opt_list = sentence_id_opt_list + sentence_add_list
            sentence_id_list.append(sentence_id_opt_list)
            label_id_list.append(label_id_opt_list)
    f_read.close()

    train_y_list = []
    with open('test_data/sentence_label.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            train_y_list.append([1, 0])

    f_read.close()
    sentence_id_numpy = np.array(sentence_id_list)
    label_id_numpy = np.array(label_id_list)
    train_y_numpy = np.array(train_y_list)

    np.save('test_data/all_sentence_id.npy', sentence_id_numpy)
    np.save('test_data/all_label_id.npy', label_id_numpy)
    np.save('test_data/all_train_y.npy', train_y_numpy)
    return

def get_test_data_batch():
    all_sentence_label = []
    with open('test_data/sentence_label.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            data_json = json.loads(data_line)
            all_sentence_label.append(data_json)
    all_sentence_numpy = np.load('test_data/all_sentence_id.npy')
    all_label_numpy = np.load('test_data/all_label_id.npy')
    all_train_y_numpy = np.load('test_data/all_train_y.npy')
    all_sentence_label_batch = []
    all_sentence_batch = []
    all_label_batch = []
    all_train_y_batch = []
    all_reward_batch = []
    sentence_label_batch = []
    sentence_batch = []
    label_batch = []
    train_y_batch = []
    reward_batch = []
    batch_num = random.randint(1, 10)
    for i, sentence in enumerate(all_sentence_numpy):
        if batch_num == 0:
            all_sentence_label_batch.append(np.array(sentence_label_batch))
            all_sentence_batch.append(np.array(sentence_batch))
            all_label_batch.append(np.array(label_batch))
            all_train_y_batch.append(np.array(train_y_batch))
            all_reward_batch.append(np.array(reward_batch))
            sentence_label_batch = []
            sentence_batch = []
            label_batch = []
            train_y_batch = []
            reward_batch = []
            batch_num = random.randint(1, 10)
        sentence_label_batch.append(all_sentence_label[i])
        sentence_batch.append(sentence.tolist())
        label_batch.append(all_label_numpy[i].tolist())
        train_y_batch.append(all_train_y_numpy[i].tolist())
        if all_train_y_numpy[i].tolist()[0] == 1:
            reward_batch.append(0.1)
        else:
            reward_batch.append(-0.1)
        batch_num = batch_num - 1
    all_sentence_label_batch_numpy = np.array(all_sentence_label_batch)
    all_sentence_batch_numpy = np.array(all_sentence_batch)
    all_label_batch_numpy = np.array(all_label_batch)
    all_train_y_batch_numpy = np.array(all_train_y_batch)
    all_reward_batch_numpy = np.array(all_reward_batch)
    np.save('test_data/sentence_label.npy', all_sentence_label_batch_numpy)
    np.save('test_data/sentence_id_batch.npy', all_sentence_batch_numpy)
    np.save('test_data/label_id_batch.npy', all_label_batch_numpy)
    np.save('test_data/train_y_batch.npy', all_train_y_batch_numpy)
    np.save('test_data/reward_batch.npy', all_reward_batch_numpy)
    return

def get_true_data():
    all_sentence_label_list = []
    with open('data/train_true.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            data_json = json.loads(data_line)
            all_sentence_label_list.append(data_json)
    f_read.close()

    with open('test_data/all_sentence_label.json', 'a', encoding='utf-8') as f_write:
        for i in range(16000,len(all_sentence_label_list)-1):
            json.dump(all_sentence_label_list[i], f_write)
            f_write.write('\n')
    f_write.close()
    return

def main():
    init_label_data()
    init_word_to_id_data()
    init_batch_data()
    get_train_test_data()
    get_test_data_id()
    get_test_data_batch()
    get_true_data()
    return

if __name__ == '__main__':
    main()