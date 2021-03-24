import numpy as np
import scipy.misc as misc

# 데이터들 배치 단위로 묶는 BatchDatset 클래스를 정의
class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0


    # 287페이지 주석 참조
    def __init__(self, records_list, image_options={}):
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()


    # raw 인풋 이미지와 annoation된 타겟 이미지를 읽음
    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        self.annotations = np.array([np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print(self.images.shape)
        print(self.annotations.shape)


    # 이미지에 변형을 가함
    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:
            image = np.array([image for i in range(3)])

        # resize옵션이 있으면 이미지 resizing을 진행
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image, [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)


    # 인풋 이미지와 타겟 이미지를 리턴
    def get_record(self):
        return self.images, self.annotations

    # batch_offset을 리셋
    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    # batch_size만큼의 다음 배치 가져옴
    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        # 한 epoch의 배치가 끝난 경우 batch index를 처음으로 다시 설정
        if self.batch_offset > self.images.shape[0]:
            # 한 epoch가 끝남
            self.epochs_completed += 1
            print("***** Epoch completed: " + str(self.epochs_completed) + "*****")
            # 데이터를 섞음 (shuffle)
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # 다음 epoch를 시작
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]


    # 전체 데이터 중에서 랜덤하게 batch_size만큼의 배치 데이터를 가져옴
    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
