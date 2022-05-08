import re
import read_sentences
import write_sentences

def sentences_without_label(sentence_list): #区分句子中有标签的和无标签的数据
    sentence_without_label = []
    sentence_with_label = []
    for sentence in sentence_list:
        label = re.findall('.stem cells \((.*?)\).', sentence)
        if len(label) == 0:
            label = re.findall('.stem cell \((.*?)\).', sentence)
        if len(label) != 0:
            sentence_with_label.append(sentence)
        else:
            sentence_without_label.append(sentence)
    return sentence_with_label, sentence_without_label

def sentence_to_label(sentence_with_label, laynum):
    json_data_list = []
    for sentence in sentence_with_label:
        json_data = {
            'sentence': '',
            'label': '',
        }
        label_list = re.findall('.stem cells \((.*?)\).', sentence)
        if len(label_list) == 0:
            label_list = re.findall('.stem cell \((.*?)\).', sentence)
        json_data['sentence'] = sentence
        sentence_list = sentence.split(' ')
        stemcell_label_list = []
        for label in label_list:
            try:
                label_index = '(' + label + ')'
                index = sentence_list.index(label_index)
                pre_label_list = re.findall('(.*?)SC.*', label)
                k = 1
                pre_label_flag = True
                for i in range(len(pre_label_list[0])-1, -1, -1):
                    if str(pre_label_list[0][i]).lower() != sentence_list[index-2-k][0].lower():
                        pre_label_flag = False
                        break
                    k += 1
                if pre_label_flag == True:
                    stem_cell_label = sentence_list[index-2] + ' ' + sentence_list[index-1]
                    for i in range(0, k-1):
                        stem_cell_label = sentence_list[index-2-i-1] + ' ' + stem_cell_label
                else:
                    stem_cell_label = sentence_list[index-3] + ' ' + sentence_list[index-2] + ' ' + sentence_list[index-1]
                stemcell_label_list.append(stem_cell_label)
            except:
                stemcell_label_list = []
        json_data['label'] = stemcell_label_list
        json_data_list.append(json_data)
        if json_data['label'] != []:
            write_sentences.main(json_data, num=laynum)
    return json_data_list

def sentence_to_index(sentence_without_label):
    sentence_index_list = []
    for sentence in sentence_without_label:
        sentence_index = {
            'sentence': '',
            'index': [],
        }
        sentence_list = sentence.split(' ')
        index_list = []
        for i,word in enumerate(sentence_list):
            if word == 'stem' or word == 'Stem':
                index_list.append(i)
        sentence_index['sentence'] = sentence
        sentence_index['index'] = index_list
        sentence_index_list.append(sentence_index)
    return sentence_index_list

def main(laynum):
    sentence_list = read_sentences.main()
    sentence_with_label,sentence_without_label = sentences_without_label(sentence_list=sentence_list)
    json_data_list = sentence_to_label(sentence_with_label=sentence_with_label, laynum=laynum)
    sentence_index_list = sentence_to_index(sentence_without_label=sentence_without_label)
    return sentence_index_list

# main(laynum=1)