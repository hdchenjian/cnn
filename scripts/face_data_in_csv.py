import cv2

all_label = {}
f = open("labels_test.txt", 'rU')
index = 0
for line in f.readlines():
   line = line.strip('\n')
   all_label[line] = index
   index += 1
f.close()

output = open("face_data_train.csv", "w")
f = open("test.txt", 'rU')
for line in f.readlines():
    img = cv2.imread(line.strip('\n'))
    label_str = line[:]
    label_index = all_label[label_str]

for i in range(n):
    image = [ord(l.read(1))]
    for j in range(28*28):
        image.append(ord(f.read(1)))
    images.append(image)

    lines = f.readlines()

test_label = []
for line in lable.readlines():
    line = line.strip('\n')

    o.write(",".join(str(pix) for pix in image)+"\n")

output.close()
lable.close()

