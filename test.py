# 텐서플로 버전확인
#import tensorflow as tf
#print(tf.__version__)


'''import os
import matplotlib.pyplot as plt

print(os.getcwd())
re = os.getcwd()
path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/training' # 폴더 경로
os.chdir(path) # 해당 폴더로 이동
files = os.listdir(path) # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음
print(files)

path_list = []
for file_name in files:
    path = os.path.join("./Data_zoo/MIT_SceneParsing/dataset/annotations/training", file_name)
    path_list.append(path)
print(path_list)

os.chdir(re)  # 다시 원래 경로로
image = plt.imread(path_list[0])
#plt.figure()
plt.imshow(image)
plt.show()

'''
import os

'''import BatchDatsetReader as dataset
IMAGE_SIZE = 224

image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
dataset.BatchDatset(train_records, image_options)

DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
print(os.path.splitext(DATA_URL.split("/")[-1])[0])'''


# ----------------------------------
# JPG이미지 변환 test
from PIL import Image
import os
'''
path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/training' # 폴더 경로
os.chdir(path) # 해당 폴더로 이동
Img = Image.open('0001train_006690.png')
width, height = Img.size
print("width  : ", width)
print("height : ", height)
print("file name : ", Img.filename)
print("format : ", Img.format_description)
#Img.save('test.jpg')
'''
# JPG이미지 변환 적용

'''path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/training' # 폴더 경로
os.chdir(path) # 해당 폴더로 이동
files = os.listdir(path) # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음

names = []
for i in files:
    n = i.replace('.png', '')
    #print(n)
    names.append(n)

print(names)  #사진파일의 이름만 저장되어있는 리스트
print(files)  #사진파일.png 리스트

for i in range(len(files)):
    Img = Image.open(files[i])
    Img.save('{}.jpg'.format(names[i]))'''



f_name = ['training', 'validation']
for f in f_name:
    path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/{}'.format(f)
    os.chdir(path)  # 해당 폴더로 이동
    files = os.listdir(path)  # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음

    names = []
    for i in files:
        n = i.replace('.png', '')
        # print(n)
        names.append(n)

    print(f, ':', names)  # 사진파일의 이름만 저장되어있는 리스트
    print(f, ':', files)  # 사진파일.png 리스트

    for i in range(len(files)):
        Img = Image.open(files[i])
        Img.save('{}.jpg'.format(names[i]))