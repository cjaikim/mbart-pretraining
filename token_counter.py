import sys
from collections import Counter


def count_tokens(encoded_files,output_file):
    token_counts = Counter()
    for f in encoded_files:
        data = open('./dataset/'+f,'r')
        for line in data:
            line = line.split()
            token_counts.update(line)

        data.close()

    with open(output_file,'w') as out_f:
        for k,c in sorted(token_counts.most_common(),key=lambda item: -item[1]):
            out_f.write('{} {}\n'.format(k,c))



if __name__=='__main__':
    encoded_files = sys.argv[2:]
    output_file = sys.argv[1]
    count_tokens(encoded_files,output_file)

    
