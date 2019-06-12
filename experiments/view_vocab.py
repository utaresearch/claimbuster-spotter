import pickle as p
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--toggle', default=False, help='Toggle b/w output and disjoint')
args = parser.parse_args()

with open('../output/vocab.pickle' if not args.toggle else '../data/disjoint_2000/vocab.pickle', 'rb') as f:
    data = p.load(f)

print(sorted(data))
