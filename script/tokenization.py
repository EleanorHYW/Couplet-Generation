import jieba

fr = open('train/out_proceeded.txt', 'rb')
fw = open('train/out_final.txt', 'wb')
while True:
	sentence = fr.readline().decode()
	if not sentence:
		break
	word_list = jieba.cut(sentence, cut_all = False)
	fw.write(' '.join(word_list).encode('utf-8'))
fr.close()
fw.close()