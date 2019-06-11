import pickle as p

with open('../output/prc_data.pickle', 'rb') as f:
    data = p.load(f)

print(data)