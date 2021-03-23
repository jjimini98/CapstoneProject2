import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# MIT Scene Parsing 데이터를 다운받을 경로
# 이거 검색창에 쳐보면 압축된 파일이 저장되는데 900메가짜리임; 어떤게 들어있는지 확인해야한다
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'

# 다운받은 MIT Scene Parsing 데이터를 읽음
def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    # MITSceneParsing.pickle 파일이 없으면 다운받은 MITSceneParsing 데이터를 pickle파일로 저장
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        SceneParsing_folder = os.path.splitext(DATA_URL.split('/')[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        print("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file!")

    # 데이터가 저장된 pickle 파일을 읽고 데이터를 training / validation 데이터로 분리
    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result
    return training_records, validation_records


# training, validation 폴더에서 raw 인풋이미지(.jpg)와 annotaion된 타켓이미지(.png)를 읽어서 리스트 형태로 만들어 리턴
# 위에 read_dataset() 안에서 작동합니다
def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found." )
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split('/')[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                #else:  #283p
                print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print('No. of % files: %d' % (directory, no_of_images))

    return image_list


