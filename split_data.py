import os
import shutil


output_path = 'watermark/'
data_path = 'watermark_original/'
paths = ['train', 'test', 'valid']

try:
    os.makedirs(output_path + 'Annotations')
except:
    print('Dir Annotations already exists')
try:
    os.makedirs(output_path + 'ImageSets/Main')
except:
    print('Dir ImageSets/Main already exists')
try:
    os.makedirs(output_path + 'JPEGImages')
except:
    print('Dir JPEGImages already exists')

for path in paths:

    xmlfilepath = os.path.join(data_path, path) + '/Annotations/'
    imagepath = os.path.join(data_path, path) + '/JPEGImages/'
    txtsavepath = output_path + '/ImageSets/Main/'
    total_xml = os.listdir(xmlfilepath)

    file = open(txtsavepath + path + '.txt', 'w')
    for line in total_xml:
        file.write(line[:-4]+'\n')
        shutil.copy(imagepath + line[:-4] + '.jpg', output_path + '/JPEGImages/')
