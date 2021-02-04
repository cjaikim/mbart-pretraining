import os
import sys
import math
import numpy as np
import argparse

def get_args():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--corpus_path',required=True)
    arg_parse.add_argument('--output_path',required=True)
    arg_parse.add_argument('--alpha',required=False,type=float,default=0.7)
    
    args = arg_parse.parse_args()
    return args

def read_data(data_dir):
    sentences = {} #holds sentences separated by language
    lang_counts = {} #number of line per language for sample scaling
    lang_id = 0 #int representing each language
    for f in os.listdir(data_dir):
        with open(data_dir+f,'r',encoding='utf-8') as data_f:
            sentences[lang_id] = []
            
            #store lines and shuffle
            [sentences[lang_id].append(line) for line in data_f]
            np.random.shuffle(sentences[lang_id])

            lang_counts[lang_id] = len(sentences[lang_id])

            lang_id += 1

    #turn lang_counts into prob proportions
    total_sentences = sum([lang_counts[lang_id] for lang_id in lang_counts])
    lang_counts = {lang_id:lang_counts[lang_id]/total_sentences for lang_id in lang_counts}

    return sentences,lang_counts

def write_sampled_set(sentences,lang_counts,alpha,output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    with open(output_dir+'sampled_set.txt','w') as sample_f:
        #calculate portion of each set to sample
        denominator = sum([lang_counts[lang_id]**alpha for lang_id in lang_counts]) #constant denominator; sum of probs^alpha
        
        #upsample and downsample
        for lang in sentences:
            ratio=(1/lang_counts[lang])*(lang_counts[lang]**alpha/denominator)
            sentences[lang] = np.random.choice(sentences[lang],math.floor(len(sentences[lang])*ratio),replace=True)
    
        #write final set to file
        for lang in sentences:
            for s in sentences[lang]:
                sample_f.write(s)
    
if __name__=='__main__':
    args = get_args()
    sentences,lang_counts = read_data(args.corpus_path)
    write_sampled_set(sentences,lang_counts,args.alpha,args.output_path)
    

