fr1 = open('train/in_proceeded.txt', 'rb')
fr2 = open('dev/in_proceeded.txt', 'rb')
fr3 = open('test/in_proceeded.txt', 'rb')
max_length = 0
max_length = 0
max_length = 0
min_length = 10
while True:
	line = fr1.readline().decode('utf-8')
	if not line:
		break;
	len1 = len(line)
	if len1 > max_length:
		max_length = len1
	if len1 < min_length:
		min_length = len1
while True:
	line = fr2.readline().decode('utf-8')
	if not line:
		break;
	len2 = len(line)
	if len2 > max_length:
		max_length = len2
	if len2 < min_length:
		min_length = len2
while True:
	line = fr3.readline().decode('utf-8')
	if not line:
		break;
	len3 = len(line)
	if len3 > max_length:
		max_length = len3
	if len3 < min_length:
		min_length = len3
print(max_length)
print(min_length)