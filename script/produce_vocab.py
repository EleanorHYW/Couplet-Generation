# -*- coding:utf-8 -*-

import json

fin1 = open('train/in_final.txt', 'rb')
fin2 = open('train/out_final.txt', 'rb')
fin3 = open('dev/in_final.txt', 'rb')
fin4 = open('dev/out_final.txt', 'rb')

vocab = {}
while True:
	line = fin1.readline()
	if not line:
		break
	words = line.split()
	for word in words:
		if word.decode('utf-8') in vocab.keys():
			continue
		else:
			vocab[word.decode('utf-8')] = len(vocab)
			print(word.decode('utf-8') + ' ' + str(vocab[word.decode('utf-8')]))

while True:
	line = fin2.readline()
	if not line:
		break
	words = line.split()
	for word in words:
		if word.decode('utf-8') in vocab.keys():
			continue
		else:
			vocab[word.decode('utf-8')] = len(vocab)

while True:
	line = fin3.readline()
	if not line:
		break
	words = line.split()
	for word in words:
		if word.decode('utf-8') in vocab.keys():
			continue
		else:
			vocab[word.decode('utf-8')] = len(vocab)

while True:
	line = fin4.readline()
	if not line:
		break
	words = line.split()
	for word in words:
		if word.decode('utf-8') in vocab.keys():
			continue
		else:
			vocab[word.decode('utf-8')] = len(vocab)

vocab['[UNK]'] = len(vocab)
vocab['[PAD]'] = len(vocab)
vocab['<s>'] = len(vocab)
vocab['</s>'] = len(vocab)

print(len(vocab))

with open("vocab_word.json","w") as f:
	json.dump(vocab, f)
	print("加载入文件完成...")