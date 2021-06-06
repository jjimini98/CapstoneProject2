###### 이건 한번만 돌려야됨

## 전체 데이터 파일 이름 가져와서 training 491 / validation 210으로 나누기 >> 무작위 추출 방법으로!!! 함수 만들어야됨
# 전체 데이터 파일 이름 가져오는데, 이때 input과 output의 사진파일 이름 차이는 마지막에 _L밖에 없음
# 그러니까 input 파일 이름만 가져와서 그거를 491, 210으로 나눠
# C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/training(나머지 210개는 validation)파일에 저장하고
# for문과 in을 사용해 output 파일도 저렇게 나눈다
# 경로는 C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/training(나머지 210은 validation)
# Labeled_data가 output / Original_data가 input


import os
import random
import shutil

'''def annotations(f_list):
    name = []
    for i in f_list:
        n = i.replace('.png', '_L.png')
        name.append(n)
    return name

path = 'C:/Users/user_/Desktop/capstone2/dataset/Original_data'
os.chdir(path)  # 해당 폴더로 이동
files = os.listdir(path)  # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음

validation = random.sample(files, 210)  # 리스트에서 val 210개 랜덤 추출 (중복 없이)
training = []  # 나머지 train 491개 파일 이름 저장리스트
for i in files:
    if i not in validation:
        training.append(i)
print('val :', len(validation), '/ train :', len(training))  # val : 210 / train : 491
#print(validation)

# annotations 함수를 이용해 .png를 _L.png로 바꿔주기
a_training = annotations(training)
a_validation = annotations(validation)
#print(a_validation)
#print(a_training)


# >> 원래 파일 경로 : 'C:/Users/user_/Desktop/capstone2/dataset/Original_data'
# training : C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/training
# validation : C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/validation
origin_path = 'C:/Users/user_/Desktop/capstone2/dataset/Original_data'
train_path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/training'
for i in training:
    shutil.copyfile(os.path.join(origin_path, i), os.path.join(train_path, i))
    # shutil.copyfile(os.path.join(원래 파일위치, 파일이름), os.path.join(파일이 복사될 위치, 파일이름))
val_path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/validation'
for i in validation:
    shutil.copyfile(os.path.join(origin_path, i), os.path.join(val_path, i))

# >> 원래 파일 경로 : 'C:/Users/user_/Desktop/capstone2/dataset/Labeled_data'
# a_training : C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/training
# a_validation : C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/validation
origin_path = 'C:/Users/user_/Desktop/capstone2/dataset/Labeled_data'
a_train_path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/training'
for i in a_training:
    shutil.copyfile(os.path.join(origin_path, i), os.path.join(a_train_path, i))
a_val_path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/validation'
for i in a_validation:
    shutil.copyfile(os.path.join(origin_path, i), os.path.join(a_val_path, i))
'''


## 같은 이미지가 잘 들어간건지 확인...
path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/images/validation'
os.chdir(path)  # 해당 폴더로 이동
image_val_files = os.listdir(path)   # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음
path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/validation'
os.chdir(path)
annotations_val_files = os.listdir(path)

names = []
for i in annotations_val_files:
    n = i.replace('_L', '')
    # print(n)
    names.append(n)

sum = 0
for i in image_val_files:
    if i in names:
        sum += 1

print(sum)  # sum 값이 210이 나왔으므로 데이터가 정상적으로 들어간것
