#### 이것도 한번만 돌리기~~~!!!
## annotations을 흑백 이미지로 바꿔봅시다~~ 절대경로 사용했으므로 사용할 때 자신의 경로에 맞춰 유의해서 기입

import cv2
import os

f_name = ['training', 'validation']
for f in f_name:
    path = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/{}'.format(f)
    os.chdir(path)  # 해당 폴더로 이동
    files = os.listdir(path)  # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음
    for i in files:
        path_ = 'C:/Users/user_/PycharmProjects/CapstoneProject2_test/Data_zoo/MIT_SceneParsing/dataset/annotations/{}/{}'.format(f, i)
        image = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(path_, image)