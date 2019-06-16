import pickle as p
import sys
sys.path.append('..')
from utils.data_loader import Dataset


with open('../data/all_data.pickle', 'rb') as f:
    data = p.load(f)

train_data = data[0]
eval_data = data[1]
vocab = data[2]

print('######################### TRAIN DATA #########################')

print([vocab[z] for z in train_data.x])
print(train_data.y)

print('######################### EVAL DATA #########################')

print([vocab[z] for z in eval_data.x])
print(eval_data.y)

print('######################### VOCAB #########################')

print(vocab)

with open('./prcdata_out.txt', 'w') as f:
    f.write(str([vocab[z] for z in train_data.x]))
    f.write('\n')
    f.write(str(train_data.y))
    f.write('\n')
    f.write(str([vocab[z] for z in eval_data.x]))
    f.write('\n')
    f.write(str(eval_data.y))
    f.write('\n')
    f.write(str(vocab))
    f.write('\n')
