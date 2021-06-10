#pycharm으로 코드를 열경우 이 파일을 사용하기 
import cv2
import numpy as np

# yolo version 3의 weights와 cfg파일 사용
YOLO_net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

# 사진에서 구별할 class 를 미리 정의 
# 우리 과제에 나온 class 사용 + 파일 내용은 sampletest.names 파일에서 확인 가능
classes = []
with open("sampletest.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLO_net에 기본으로 가지고 있던 layer의 이름들을 변수 layers_names에 저장 
layers_names = YOLO_net.getLayerNames()
# layers_names 출력  결과(Yolo 알고리즘의 레이어) : ['conv_0', 'bn_0', 'relu_1', 'pool_1', 'conv_2', 'bn_2', 'relu_3', 'pool_3', 'conv_4', 'bn_4', 'relu_5', 'pool_5', 'conv_6', 'bn_6', 'relu_7', 'pool_7', 'conv_8', 'bn_8', 'relu_9', 'pool_9', 'conv_10', 'bn_10', 'relu_11', 'pool_11', 'conv_12', 'bn_12', 'relu_13', 'conv_13', 'bn_13', 'relu_14', 'conv_14', 'permute_15', 'detection_out']

# output_layers의 출력결과 :  연결되지 않은 layer의 인덱스를 반환하는 함수 yolo 알고리즘에서 제일 마지막 단(출력이 되는) 만 따로 모아 output_layer로 저장
output_layers = [layers_names[i[0]-1] for i in YOLO_net.getUnconnectedOutLayers()]


# 난수 생성(난수의 범위 : 0~255) : 라벨링된 결과를 출력할 때 쓸 색깔 변수
colors = np.random.uniform(0,255, size = (len(classes),3))

#이미지 읽어들이기
img = cv2.imread("test.png")
# img = cv2.resize(img, None, fx = 0.4 , fy = 0.4) 이렇게 하면 이미지가 너무 작아져서 resize는 하지 않음 이미지 사이즈를 축소하는 코드
# 이미지의 크기를 각각 변수에 저장
height , width , channels = img.shape

#객체인식 시작
#이미지를 YOLO_net에 넣기 위해서 4차원의 이미지로 변경해야한다.
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True , crop  = False)
YOLO_net.setInput(blob)

# outs에는 yolo로 학습한 객체인식 결과 및 신뢰도 등이 담겨져 있음
outs = YOLO_net.forward(output_layers)

#이미지 정보를 보여주기
class_ids = []
confidences = []
boxes = []
for out in outs:
    
    for detection in out:
        scores = detection[5:] #객체 인식의 결과
        class_id = np.argmax(scores)   #전체 score 중 가장 큰 값의 인덱스를 class_id로 저장
        confidence = scores[class_id] # score에서 제일 큰 값을 confidence로 저장
        if confidence > 0.5: # confidence 값이 0.5를 넘은 경우 객체 인식 결과로 신뢰도가 생김
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2) #해당하는 바운딩 박스를 그림
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)



indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #여러 바운딩 박스 중 가장 객체를 잘 인식하는 박스를 찾는 과정 / 0.5는 score confidence, 0.4는 nms confidence


font = cv2.FONT_HERSHEY_PLAIN #결과로 보여질 라벨의 폰트 지정

for i in range(len(boxes)): # 바운딩 박스 생성
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

# 이미지를 새 창으로 보여주고 싶으면 밑의 코드 3줄 실행         
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

