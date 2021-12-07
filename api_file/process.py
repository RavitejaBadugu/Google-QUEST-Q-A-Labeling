import numpy as np
from transformers import BertTokenizer,RobertaTokenizer,XLNetTokenizer


def Tokenizer(PRE_NAME,tokenizer_path):
    if PRE_NAME.startswith('bert'):
        tokenizer=BertTokenizer.from_pretrained(tokenizer_path)
    elif PRE_NAME.startswith('roberta'):
        tokenizer=RobertaTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer=XLNetTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def get_bert_single_head_inputs(tokenizer,s1,s2,max_length):
    s1=tokenizer.tokenize(s1)
    s2=tokenizer.tokenize(s2)
    s_len=max_length-3
    s1_len=len(s1)
    s2_len=len(s2)
    if s1_len+s2_len==s_len:
        total_tokens=['[CLS]']+s1+['[SEP]']+s2+['[SEP]']
    elif s1_len+s2_len<s_len:
        total_tokens=['[CLS]']+s1+['[SEP]']+s2+['[SEP]']
    else:
        if s_len%2==0: #even length so, lets divide equally
            req_s1_len=s_len//2
            req_s2_len=s_len//2
        else:
            req_s1_len=s_len//2
            req_s2_len=(s_len//2)+1
        if s1_len<=req_s1_len and s2_len>req_s2_len:
            # s1 is shorter but s2 is longer
            s2=s2[:s_len-s1_len]
            total_tokens=['[CLS]']+s1+['[SEP]']+s2+['[SEP]']
        elif s1_len>req_s1_len and s2_len<=req_s2_len:
            # s1 is longer but s2 is shorter
            s1=s1[:s_len-s2_len]
            total_tokens=['[CLS]']+s1+['[SEP]']+s2+['[SEP]']
        elif s1_len>req_s1_len and s2_len>req_s2_len:
            # both are longer
            s1=s1[:req_s1_len]
            s2=s2[:req_s2_len]
            total_tokens=['[CLS]']+s1+['[SEP]']+s2+['[SEP]']
    total_tokens=total_tokens+['[PAD]']*(max_length-len(total_tokens))
    input_ids=tokenizer.convert_tokens_to_ids(total_tokens)
    attention_mask=np.char.not_equal(total_tokens,'[PAD]').astype('int32')
    token_type_ids=[]
    seq_num=0
    for tok in total_tokens:
        if tok=='[SEP]':
            token_type_ids.append(seq_num)
            seq_num=1-seq_num
        else:
            token_type_ids.append(seq_num)
    return input_ids,attention_mask,token_type_ids


def get_roberta_single_head_inputs(tokenizer,s1,s2,max_length):
    s1=tokenizer.tokenize(s1)
    s2=tokenizer.tokenize(s2)
    s_len=max_length-4
    s1_len=len(s1)
    s2_len=len(s2)
    if s1_len+s2_len==s_len:
        total_tokens=['<s>']+s1+['</s>']+['</s>']+s2+['</s>']
    elif s1_len+s2_len<s_len:
        total_tokens=['<s>']+s1+['</s>']+['</s>']+s2+['</s>']
    else:
        if s_len%2==0: #even length so, lets divide equally
            req_s1_len=s_len//2
            req_s2_len=s_len//2
        else:
            req_s1_len=s_len//2
            req_s2_len=(s_len//2)+1
        if s1_len<=req_s1_len and s2_len>req_s2_len:
            # s1 is shorter but s2 is longer
            s2=s2[:s_len-s1_len]
            total_tokens=['<s>']+s1+['</s>']+['</s>']+s2+['</s>']
        elif s1_len>req_s1_len and s2_len<=req_s2_len:
            # s1 is longer but s2 is shorter
            s1=s1[:s_len-s2_len]
            total_tokens=['<s>']+s1+['</s>']+['</s>']+s2+['</s>']
        elif s1_len>req_s1_len and s2_len>req_s2_len:
            # both are longer
            s1=s1[:req_s1_len]
            s2=s2[:req_s2_len]
            total_tokens=['<s>']+s1+['</s>']+['</s>']+s2+['</s>']
    total_tokens=total_tokens+['<pad>']*(max_length-len(total_tokens))
    input_ids=tokenizer.convert_tokens_to_ids(total_tokens)
    attention_mask=np.char.not_equal(total_tokens,'<pad>').astype('int32')
    return input_ids,attention_mask


def get_xlnet_single_head_inputs(tokenizer,s1,s2,max_length):
    s1=tokenizer.tokenize(s1)
    s2=tokenizer.tokenize(s2)
    s_len=max_length-3
    s1_len=len(s1)
    s2_len=len(s2)
    if s1_len+s2_len==s_len:
        total_tokens=s1+['<sep>']+s2+['<sep>']+['<cls>']
    elif s1_len+s2_len<s_len:
        total_tokens=s1+['<sep>']+s2+['<sep>']+['<cls>']
    else:
        if s_len%2==0: #even length so, lets divide equally
            req_s1_len=s_len//2
            req_s2_len=s_len//2
        else:
            req_s1_len=s_len//2
            req_s2_len=(s_len//2)+1
        if s1_len<=req_s1_len and s2_len>req_s2_len:
            # s1 is shorter but s2 is longer
            s2=s2[:s_len-s1_len]
            total_tokens=s1+['<sep>']+s2+['<sep>']+['<cls>']
        elif s1_len>req_s1_len and s2_len<=req_s2_len:
            # s1 is longer but s2 is shorter
            s1=s1[:s_len-s2_len]
            total_tokens=s1+['<sep>']+s2+['<sep>']+['<cls>']
        elif s1_len>req_s1_len and s2_len>req_s2_len:
            # both are longer
            s1=s1[:req_s1_len]
            s2=s2[:req_s2_len]
            total_tokens=s1+['<sep>']+s2+['<sep>']+['<cls>']
    token_type_ids=[]
    seq_num=0
    for tok in total_tokens:
        if tok=='<sep>':
            token_type_ids.append(seq_num)
            seq_num=1-seq_num
        elif tok=='<cls>':
            token_type_ids.append(2)
        else:
            token_type_ids.append(seq_num)
    token_type_ids=[3]*(max_length-len(total_tokens))+token_type_ids
    total_tokens=['<pad>']*(max_length-len(total_tokens))+total_tokens
    input_ids=tokenizer.convert_tokens_to_ids(total_tokens)
    attention_mask=np.char.not_equal(total_tokens,'<pad>').astype('int32')
    return input_ids,attention_mask,token_type_ids