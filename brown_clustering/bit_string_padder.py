# Makes Bit Strings to Similar Length
__author__ = "Chirayu Desai"

data = {}
with open('strings.txt', 'r') as file:
    lines = file.readlines()
    lines = [eval(line.strip()) for line in lines]
    for line in lines:
        data[line[0]] = line[1]


max_length = max(map(len, data.values()))

for key in data.keys():
    data[key] = data[key].zfill(max_length)

f = open('padded_bit_strings.txt', 'w')
for item in data.items():
    f.write(str(item))
    f.write('\n')
f.close()
