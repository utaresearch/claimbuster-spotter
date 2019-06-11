import pickle as p

with open('../output/vocab.pickle', 'rb') as f:
    data = p.load(f)

print(data)