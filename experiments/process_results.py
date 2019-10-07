data = []

with open('idir_results.txt', 'r') as f:
	f.readline()
	data.append(f.readline())
	for count, line in enumerate(f):
		if count % 3 == 1:
			data.append(line)

extracted_data = []

for line in data:
	search_string = 'Loss:   '
	idx = line.find(search_string)
	loss = float(line[idx+len(search_string):idx+5+len(search_string)])

	search_string = 'Dev F1:  '
	idx = line.find(search_string)
	f1 = float(line[idx+len(search_string):idx+6+len(search_string)])

	extracted_data.append((f1, loss))

extracted_data = sorted(list(reversed(sorted(extracted_data)))[:10], key=lambda x:x[1])
print(extracted_data)