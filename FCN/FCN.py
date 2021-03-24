from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime

import TensorflowUtils as utils
import read_SceneParsingData as scene_parsing
import BatchDatsetReader as dataset

# 학습에 필요한 설정값들을 tf.flag.FLAGS로 지정
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string('mode', "train", "Mode train/ visualize")

# VGG-19의 파라미터가 저장된 mat 파일(MATLAB 파일)을 받아올 경로 지정
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

# 학습에 필요한 설정값들 지정
MAX_ITERATION = int(100000 + 1)
NUM_OF_CLASSESS = 151 #레이블 개수
IMAGE_SIZE = 224


# VGGNet 그래프 구조 구축
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3'

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4'
                                         
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        # 컨볼루션층일 경우
        if kind == 'convolution':
            kernels, bias = weights[1][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # MATLAB 파일의 행렬 순서를 tensorflow 행렬의 순서로 변환
            kernels = utils.get_variable(np.transpose(kernels, (1,0,2,3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        # Activation층일 경우
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        # 풀링층일 경우
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


# FCN 그래프 구조를 정의
def inference(image, keep_prob):
    # arguments:
    # image: 인풋 이미지 0-255 사이의 값을 가지고 있어야 함
    # keep_prob: 드롭아웃에서 드롭하지 않을 노드의 비율

    # 다운받은 VVGNet을 불러온다
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    # 이미지에 Mean Normalization을 수행
    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        # VGGNet의 conv5(conv5_3) 레이어를 불러옴
        conv_final_layer = image_net["conv5_3"]

        # pool5 정의
        pool5 = utils.max_pool_2x2(conv_final_layer)

        # conv6 정의
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        # conv7 정의 (1x1 conv)
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        # conv8 정의 (1x1 conv)
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

        # FCN-8s를 위한 Skip Layers Fusion 설정
        # 이제 원본 이미지 크기로 Upsampling하기 위한 deconv 레이어 정의
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # conv8의 이미지 2배 확대
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        # 2x conv8과 pool4를 더해 fuse_1 이미지 만들기
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        # fuse_1의 이미지 2배 확대
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        # 2x fuse_1과 pool3를 더해 fuse_2 이미지 만들기
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        # fuse_2의 이미지 8배 확대
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        # 최종 prediction 결과를 결정하기 위해 마지막 activation들 중에서
        # argmax로 최대값을 가진 activation을 추출
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


# 263p~~~~~~~~~~~~~~~~