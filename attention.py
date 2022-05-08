from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel


def get_attention(sentence_a, layer_num):
    model_version = 'biobert_base_cased'
    model = BertModel.from_pretrained(model_version)
    tokenizer = BertTokenizer.from_pretrained(model_version)
    inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_all = model(input_ids, token_type_ids=token_type_ids, output_attentions=True)[-1]
    attention = attention_all[layer_num]
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    return attention,tokens

def main(sentence_a, laynum):
    layer_num = laynum #获取bert模型中的第layer_num层attention矩阵(0~12)
    attention,tokens = get_attention(sentence_a=sentence_a, layer_num=layer_num)
    return attention,tokens

