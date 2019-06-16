import pickle as p


with open('../data/all_data.pickle', 'rb') as f:
    data = p.load(f)

train_data = data[0]
eval_data = data[1]
vocab = data[2]

print('######################### TRAIN DATA #########################')

print(train_data.x)
print(train_data.y)

print('######################### EVAL DATA #########################')

print(eval_data.x)
print(eval_data.y)

print('######################### VOCAB #########################')

print(vocab)