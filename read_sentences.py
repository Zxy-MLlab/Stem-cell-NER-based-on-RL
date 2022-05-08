import re
import json

def read_sentences(flag):
    sentence_list = []
    if flag == 'train':
        with open('gule_data/train.txt', 'r', encoding='utf-8') as f_read:
            data_lines = f_read.readlines()
        f_read.close()
        for data in data_lines:
            sentence_list.append(data.replace('\n',''))
    else:
        with open('gule_data/all_sentence_label1.json', 'r', encoding='utf-8') as f_read:
            while True:
                data_lines = f_read.readline()
                if not data_lines:
                    break
                json_data = json.loads(data_lines)
                sentence = json_data['sentence']
                sentence_list.append(sentence)
        f_read.close()
    return  sentence_list

def main():
    flag = 'devel'
    sentence_list = read_sentences(flag=flag)
    return sentence_list

main()