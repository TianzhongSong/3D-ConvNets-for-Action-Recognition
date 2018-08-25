import os

img_path = '/home/deep/datasets/hmdb/'

clip_length = 16

f = open('ucfTrainTestlist/hmdb.txt', 'w')

actions = os.listdir(img_path)
actions.sort(key=str.lower)
label = 0
for action in actions:
    print(action)
    samples = os.listdir(img_path + action)
    samples.sort(key=str.lower)
    for sample in samples:
        f.write(action + '/' + sample + ' ' + str(label) + '\n')
    label += 1
f.close()
