import predata
import attention
import rearch
import write_sentences
import json
import tqdm

def main(laynum):
    sentence_index_list = predata.main(laynum=laynum)
    sentence_label_list = []
    with open('output/sentence_label.json', 'a', encoding='utf-8') as f_write:
        for sentence_index_temp in tqdm.tqdm(sentence_index_list):
            stemcell_label = {
                'sentence':'',
                'label':'',
            }
            try:
                sentence_a = sentence_index_temp['sentence']
                sentence_index = sentence_index_temp['index']
                attention_matrix,tokens = attention.main(sentence_a=sentence_a, laynum=laynum)
                stem_cell_label = rearch.main(attention_matrix=attention_matrix, tokens=tokens, sentence_a=sentence_a,sentence_index_list=sentence_index)
                stemcell_label['sentence'] = sentence_a
                stemcell_label['label'] = stem_cell_label
                sentence_label_list.append(stemcell_label)
                write_sentences.main(sentence_label_json=stemcell_label, num=laynum)
            except:
                continue
    return

if __name__ == '__main__':
    main(laynum=3)