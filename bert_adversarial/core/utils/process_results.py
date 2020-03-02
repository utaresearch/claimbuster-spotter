# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#

data = []

with open('idir_results.txt', 'r') as f:
	f.readline()
	data.append(f.readline())
	for count, line in enumerate(f):
		if count % 3 == 1:
			data.append(line)

extracted_data = []

for line in data:
	search_string = 'Dev Loss: '
	idx = line.find(search_string)
	loss = float(line[idx+len(search_string):idx+7+len(search_string)])

	search_string = 'Dev F1:  '
	idx = line.find(search_string)
	f1 = float(line[idx+len(search_string):idx+6+len(search_string)])

	extracted_data.append((f1, loss))

extracted_data = sorted(list(reversed(sorted(extracted_data)))[:10], key=lambda x:x[1])
print(extracted_data)