mapping = {}

with open("emnist-balanced-mapping.txt") as f:
    for line in f:
        label, ascii_code = line.split()
        mapping[int(label)] = chr(int(ascii_code))

print(emn)