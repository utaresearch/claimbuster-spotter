import pickle
import json


if __name__ == '__main__':
    with open('./disjoint_2000.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data)
        with open('./disjoint_2000.json', 'w') as fout:
            json.dump(data, fout)