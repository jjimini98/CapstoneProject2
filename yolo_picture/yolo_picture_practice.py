import cv2
import numpy as np

# yolo 불러오기
YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")

# 사진에서 구별할 class 를 미리 정의 
classes = []
with open("sampletest.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 
layers_names = YOLO_net.getLayerNames()
output_layers = [layers_names[i[0]-1] for i in YOLO_net.getUnconnectedOutLayers()]

# 객체 인식 후 그려질 박스의 색깔과 사이즈 지정
colors = np.random.uniform(0,255, size = (len(classes),3))

#이미지 로드
img = cv2.imread("sample.png")
img = cv2.resize(img, None, fx = 0.4 , fy = 0.4)
height , width , channels = img.shape

#객체인식 시작
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True , crop  = False)
YOLO_net.setInput(blob)
outs = YOLO_net.forward(output_layers)

#이미지 정보를 보여주기
class_ids = []
confidences = []
boxes = []
for out in outs:
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

        
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

