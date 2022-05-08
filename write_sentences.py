import re
import json

def write_data(sentence_label_json, num):
    path = 'output/sentence_label' + '(' + str(num) + ')'+'.json'
    with open(path, 'a', encoding='utf-8') as f_write:
        json.dump(sentence_label_json, f_write)
        f_write.write('\n')
    return

def main(sentence_label_json, num):
    write_data(sentence_label_json=sentence_label_json, num=num)
    return