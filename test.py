import json
import os

img_path = '/media/ri2raj/External HDD/Task01_BrainTumour/train/'
test_img_path = '/media/ri2raj/External HDD/Task01_BrainTumour/test/'
label_path = '/media/ri2raj/External HDD/Task01_BrainTumour/train/'
test_label_path = '/media/ri2raj/External HDD/Task01_BrainTumour/test/'
file = open('/media/ri2raj/External HDD/Task01_BrainTumour/dataset.json')
data = json.load(file)
img = []
label = []

for j in data['test']:
    x = j['image']
    y = j['label']
    if x.startswith('./'):
        x = x[2:]
        img.append(x)
        img.sort()
    if y.startswith('./'):
        y = y[2:]
        label.append(y)
        label.sort()
file.close()


for i in range(len(img)):
    print(img[i])