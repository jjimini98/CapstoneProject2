import cv2

YOLO_net = cv2.dnn.readNet("./yolo_picture/yolov2-tiny.weights","./yolo_picture/yolov2-tiny.cfg")
layers_names = YOLO_net.getLayerNames()



img = cv2.imread("./yolo_picture/sample.png")
height , width , channels = img.shape
print(height , width, channels)
print(img.shape)
# img = cv2.resize(img, None, fx = 0.4 , fy = 0.4)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
