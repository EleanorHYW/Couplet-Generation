import re

fr = open('dev/in.txt', 'rb')
fw = open('dev/in_proceeded.txt', 'wb')
while True:
	line = fr.readline().decode('utf-8')
	if not line:
		break
	line = ''.join(line.split()) + '\n'
	fw.write(line.encode('utf-8'))
fr.close()
fw.close()

fr = open('dev/out.txt', 'rb')
fw = open('dev/out_proceeded.txt', 'wb')
while True:
	line = fr.readline().decode('utf-8')
	if not line:
		break
	line = ''.join(line.split()) + '\n'
	fw.write(line.encode('utf-8'))
fr.close()
fw.close()
