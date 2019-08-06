import json

with open('data_small.json', 'r') as f:
    train_data = json.load(f)
with open('disjoint_2000.json', 'r') as f:
    test_data = json.load(f)

train_info = [sum(z['label'] == str(i) for z in train_data) for i in [-1, 0, 1]]
test_info = [sum(z['label'] == i for z in test_data) for i in [-1, 0, 1]]
print(train_info)
print(test_info)

sum_train = sum(train_info)
sum_test = sum(test_info)
train_div = [z / sum_train for z in train_info]
test_div = [z / sum_test for z in test_info]

print(train_div)
print(test_div)