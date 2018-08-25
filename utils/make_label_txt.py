import argparse
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate label txt for ucf101')
    parser.add_argument('--image-path', type=str, default='', help='image path')
    args = parser.parse_args()

    f1 = open('ucfTrainTestlist/train_file.txt', 'r')
    f2 = open('ucfTrainTestlist/test_file.txt', 'r')

    train_list = f1.readlines()
    test_list = f2.readlines()

    f3 = open('train_list.txt', 'w')
    f4 = open('test_list.txt', 'w')

    clip_length = 16

    for item in train_list:
        name = item.split(' ')[0]
        image_path = args.image_path+name
        label = item.split(' ')[-1]
        images = os.listdir(image_path)
        nb = len(images) // clip_length
        if len(images) % clip_length < 8:
            nb -= 1
        for i in range(nb):
            f3.write(name + ' ' + str(i*clip_length+1)+' '+label)

    for item in test_list:
        name = item.split(' ')[0]
        image_path = args.image_path+name
        label = item.split(' ')[-1]
        images = os.listdir(image_path)
        nb = len(images) // clip_length
        if len(images) % clip_length < 8:
            nb -= 1
        for i in range(nb):
            f4.write(name + ' ' + str(i*clip_length+1)+' '+label)

    f1.close()
    f2.close()
    f3.close()
    f4.close()
