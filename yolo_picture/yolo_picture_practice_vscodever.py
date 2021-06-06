#vscode로 코드를 열경우 이 파일을 사용하기 

import cv2
import numpy as np

# yolo 불러오기
YOLO_net = cv2.dnn.readNet("./yolo_picture/yolov2-tiny.weights","./yolo_picture/yolov2-tiny.cfg")

# 사진에서 구별할 class 를 미리 정의 
# 우리 과제에 나온 class 사용 + 파일 내용은 sampletest.names 파일에서 확인 가능
classes = []
with open("./yolo_picture/sampletest.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# YOLO_net에 기본으로 가지고 있던 layer의 이름들을 변수 layers_names에 저장 
layers_names = YOLO_net.getLayerNames()
# layers_names 출력  결과 : ['conv_0', 'bn_0', 'relu_1', 'pool_1', 'conv_2', 'bn_2', 'relu_3', 'pool_3', 'conv_4', 'bn_4', 'relu_5', 'pool_5', 'conv_6', 'bn_6', 'relu_7', 'pool_7', 'conv_8', 'bn_8', 'relu_9', 'pool_9', 'conv_10', 'bn_10', 'relu_11', 'pool_11', 'conv_12', 'bn_12', 'relu_13', 'conv_13', 'bn_13', 'relu_14', 'conv_14', 'permute_15', 'detection_out']

# output_layers의 출력결과 : ['detection_out'] YOLO_net.get ,  연결되지 않은 layer의 인덱스를 반환하는 함수 
output_layers = [layers_names[i[0]-1] for i in YOLO_net.getUnconnectedOutLayers()]


# 11행 3열 사이즈의 난수 생성(난수의 범위 : 0~255) 
colors = np.random.uniform(0,255, size = (len(classes),3)) 



#이미지 로드
img = cv2.imread("./yolo_picture/sample.png")
# img = cv2.resize(img, None, fx = 0.4, fy = 0.4) #이렇게 하면 이미지가 너무 작아져서 resize는 하지 않음
height , width , channels = img.shape

#객체인식 시작
#이미지를 YOLO_net에 넣기 위해서 4차원의 이미지로 변경해야함. 
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True , crop  = False)
YOLO_net.setInput(blob)

# 위에서 정의한 output_layers의 형태로 output blob 값을 return  ??? 
outs = YOLO_net.forward(output_layers)


#이미지 정보를 보여주기
class_ids = [] #[9]
confidences = [] #[0.7555906176567078]
boxes = [] #[[252, 271, 36, 42]]

for out in outs:  # 배열의 한 행씩 가지고 나온다. 
    for detection in out:

        scores = detection[5:]
        class_id = np.argmax(scores)  
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)



indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
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

