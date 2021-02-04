import os
import argparse
import json

from transformers import MBartModel,MBartConfig,MBartTokenizer
#import transformers

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path',required=True)
    arg_parser.add_argument('--output_path',required=True)
    arg_parser.add_argument('--model_file',required=True)

    args = arg_parser.parse_args()
    return args

def train_MBart(data_path,tokenizer,output_path):
    model_config = MBartConfig(vocab_size=300,d_model=10,encoder_layers=1,decoder_layers=1,encoder_attention_heads=1,decoder_attention_heads=1,encoder_ffn_dim=10,decoder_ffn_dim=10,max_position_embeddings=512)
    model = MBartModel(config=model_config)

    sentences = {} #associates lang_id with list of sentences
    
    #read data files and separate language data into different lists
    lang_id = 0 #counter for languages in dataset
    for sentence_file in os.listdir(data_path):
        with open(data_path+sentence_file,'r') as data:
            sentences[lang_id] = []
            for line in data:
                sentences[lang_id].append(line)
        lang_id += 1

    #create token sequences to pass into model
    src_lang,tgt_lang = (sentences[lang_id] for lang_id in sentences)
    batch = tokenizer.prepare_seq2seq_batch(src_texts=src_lang,tgt_texts=tgt_lang,return_tensors='pt')
    
    #find max sequence len
    max_seq_len = 0
    for seq in batch['input_ids']:
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
    print(max_seq_len)
    
    model(input_ids=batch['input_ids'][:5],decoder_input_ids=batch['labels'][:5])
    model.save_pretrained(output_path)
            
if __name__=='__main__':
    args = get_args()
    
    tokenizer = MBartTokenizer(vocab_file=args.model_file) #create new MBTokenizer from file created by google/sentencepiece
    
    train_MBart(args.data_path,tokenizer,args.output_path)

