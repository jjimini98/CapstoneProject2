# 모델 구현을 위한 유틸리티 함수들
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io

# VGGNet 파라미터가 저장된 mat 파일을 다운로드 받고 불러옵니다
def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!!!")
    data = scipy.io.loadmat(filepath)
    return data

# dir_path에 url_name에서 다운받은 zip파일의 압축을 해제
# 아마 이 코드는 안쓸듯 하다?! 우리는 그냥 폴더를 아예 주니까요... 일단 씁니다 이 함수를 위에 get_model_data에서 사용했을때 어떻게 불러와지는지 확인해야합니다.....
def maybe_download_and_extract(dir_path, url_name, is_tarfile = False, is_zipfile = False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.join(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>>Downloding %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

# 이미지를 png파일로 저장
def save_image(image, save_dir, name, mean=None):
    if mean:    # 만약 평균값을 argument로 받으면 평균값을 더한 뒤에 이미지를 저장하고, 아니면 바로 이미지를 저장
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + '.png'), image)

# 변수 선언
def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var

# weight 선언
def weigth_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

# bias 선언
def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

# 컨볼루션 정의
def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

# 디컨볼루션(Transpose Convolution)을 정의
def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    #print x.get_shape()
    #print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    #print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,stride,stride,1], padding="SAME")
    return tf.nn.bias_add(conv, b)

# 2x2 max 풀링 정의
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# 2x2 average 풀링 정의
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# 이미지에 평균 빼기
def process_image(image, mean_pixel):
    return image - mean_pixel

# 이미지에 평균 더하기 >> save_image()에서 사용
def unprocess_image(image, mean_pixel):
    return image + mean_pixel