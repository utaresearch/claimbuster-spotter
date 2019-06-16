import pickle as p
import sys
sys.path.append('..')
from utils.data_loader import Dataset


with open('../data/all_data.pickle', 'rb') as f:
    data = p.load(f)

train_data = data[0]
eval_data = data[1]
vocab = data[2]
inv_vocab = {v: k for k, v in vocab.items()}


train_data_x = [' '.join([inv_vocab[z] for z in sentence]) for sentence in train_data.x]
eval_data_x = [' '.join([inv_vocab[z] for z in sentence]) for sentence in eval_data.x]

print('######################### TRAIN DATA #########################')

print(train_data_x)
print(train_data.y)

print('######################### EVAL DATA #########################')

print(eval_data_x)
print(eval_data.y)

print('######################### VOCAB #########################')

print(vocab)

with open('./prcdata_out.txt', 'w') as f:
    f.write(str(train_data_x))
    f.write('\n')
    f.write(str(train_data.y))
    f.write('\n')
    f.write(str(eval_data_x))
    f.write('\n')
    f.write(str(eval_data.y))
    f.write('\n')
    f.write(str(vocab))
    f.write('\n')
