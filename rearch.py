import re
import numpy as np
from transformers import BertTokenizer, BertModel

def get_word_attention(attention_matrix, index):
    word_attention_matrix = attention_matrix[index]
    return word_attention_matrix


def tokens_to_index(tokens):
    index_list = []
    for i,word in enumerate(tokens):
        if word == 'stem' or word == 'Stem':
            index_list.append(i)
    return index_list

def attention_to_softmax(attention_matrix):
    attention_matrix = attention_matrix[0][0].detach().numpy()
    softmax_attention_matrix = []
    row_num = attention_matrix.shape[0]  # 获取attention矩阵的行数
    value_row_max = attention_matrix.max(axis=1).reshape(row_num, 1)  # 计算每行的最大值，转换成维度相同的矩阵
    attention_without_max = attention_matrix - value_row_max  # 每行元素减去最大值
    attention_matrix_exp = np.exp(attention_without_max)
    value_row_sum = attention_matrix_exp.sum(axis=1).reshape(row_num, 1)  # 计算每行元素和
    attention_softmax_opt = attention_matrix_exp / value_row_sum  # 计算softmax值
    softmax_attention_matrix.append(attention_softmax_opt)
    softmax_attention_matrix = softmax_attention_matrix[0]
    return softmax_attention_matrix

def rearch_word(softmax_attention_matrix, tokens_index, tokens, sentence_a, sentence_index):
    stem_cell_label = tokens[tokens_index] + ' ' + tokens[tokens_index+1].replace(',','').replace('.','')
    sentence_list = sentence_a.split(' ')
    threshold = 1.0 / len(tokens)  #根据句子序列的长度选择阈值
    word_attention_matrix = get_word_attention(attention_matrix=softmax_attention_matrix, index=tokens_index)
    threshold_value = word_attention_matrix[tokens_index-1] #stem前一个词
    while True:
        if threshold_value < threshold:
            break
        else:
            if sentence_index-1 < 0:
                break
            stem_cell_label = sentence_list[sentence_index-1] + ' ' + stem_cell_label
            model_version = 'biobert_base_cased'
            tokenizer = BertTokenizer.from_pretrained(model_version)
            word_tokens = tokenizer.tokenize(sentence_list[sentence_index-1])
            word_token_lens = len(word_tokens)
            tokens_index = tokens_index - word_token_lens
            sentence_index = sentence_index - 1
            word_attention_matrix = get_word_attention(attention_matrix=softmax_attention_matrix, index=tokens_index)
            threshold_value = word_attention_matrix[tokens_index-1]
            pass
    return stem_cell_label

def main(attention_matrix, tokens, sentence_a, sentence_index_list):
    index_list = tokens_to_index(tokens=tokens)
    stemcell_label_list = []
    for i,index in enumerate(index_list):
        softmax_attention_matrix = attention_to_softmax(attention_matrix=attention_matrix)
        try:
            sentence_index = sentence_index_list[i]
        except:
            break
        stem_cell_label = rearch_word(softmax_attention_matrix=softmax_attention_matrix,tokens_index=index,tokens=tokens,sentence_a=sentence_a,sentence_index=sentence_index)
        stemcell_label_list.append(stem_cell_label)
    return stemcell_label_list