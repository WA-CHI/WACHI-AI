import cv2
import numpy as np
import time

# YOLOv3 관련 파일 경로
weights_path = 'yolov3_training_last.weights'
config_path = 'yolov3_testing.cfg'
labels_path = 'coco.names'

# 클래스 레이블 로드
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

# YOLOv3 모델 로드
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
output_layers = net.getUnconnectedOutLayersNames()

# 웹캠 스트리밍 시작
cap = cv2.VideoCapture(0)  # 0은 내장 웹캠을 나타냅니다.
if not cap.isOpened():
    print("camera open failed")

count = 0
recording = False  # Initialize recording as False

# 감지 박스 크기 조절을 위한 변수 설정
box_scale = 1.0  # 감지 박스 크기 조절 비율

record_duration = 10  # 녹화 지속 시간 (초)
video_counter = 1  # 비디오 카운터

while True:
    ret, frame = cap.read()
    
    if count == 2:
        if not recording:
            start_time = time.time()  # 녹화 시작 시간 기록
            recording = True  # 녹화 시작 상태로 변경
            video_name = f'video_{video_counter}.mp4'
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame.shape[1], frame.shape[0]))

        if recording:
            out.write(frame)  # 프레임 저장

        # 녹화가 시작된 후 record_duration(10초)가 지나면 녹화 종료
        if recording and time.time() - start_time >= record_duration:
            out.release()  # 녹화 종료
            recording = False  # 녹화 상태 초기화
            count = 0  # 녹화 종료 후 카운트 초기화
            video_counter += 1  # 비디오 카운터 증가

    # 프레임 전처리
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # 객체 검출
    layer_outputs = net.forward(output_layers)

    # 검출된 객체 정보 저장
    boxes = []
    confidences = []
    class_ids = []

    # 검출된 객체 정보 추출
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8:  # confidence threshold 설정
                count = count + 1
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1] * box_scale)
                height = int(detection[3] * frame.shape[0] * box_scale)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
   
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 겹치는 경계 상자 제거
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 검출된 객체 시각화
    for i in indices:
        i = i.item()  # 객체 인덱스를 스칼라 변수로 가져옵니다.
        x, y, width, height = boxes[i]
        label = labels[class_ids[i]]
        confidence = confidences[i]

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow('YOLOv3 Object Detection', frame)

    # 종료 키 확인
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
