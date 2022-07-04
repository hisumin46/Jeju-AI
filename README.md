## jeju_AI
미래신산업 수요특화형 AI 교육 대회 프로젝트입니다.

5일동안 수업받은 내용을 토대로 정리하였고, 프로젝트를 진행하였습니다.
## 최종 프로젝트
- 얼굴 인식 드론
### 데이터 가공
```python
def process_data(path):
  x = []
  y = []
  dir_list = os.listdir(path)
  for i in range(0,len(dir_list)):
    dir_path = path + "/" + dir_list[i]
    dir_name = os.listdir(dir_path)
    
    for j in range(len(dir_name)):
      full_dir_path = dir_path + "/" + dir_name[j]
      img = cv2.imread(full_dir_path,cv2.IMREAD_COLOR)
      
      # HSV 색 변경 
      img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      
      # 필터 fastNlMeansDenoisingColored 사용하여 이미지 가공
      img_hsv = cv2.fastNlMeansDenoisingColored(img_hsv,None,10,10,7,21)
      lower = np.array([0,48,80], dtype="uint8")
      upper = np.array([20,255,255], dtype="uint8")
      img_hand = cv2.inRange(img_hsv,lower,upper)
      
      #경계선 찾음
      contours, hierarchy = cv2.findContours(img_hand, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      # 가장 큰 영역 찾기
      max = 0
      maxcnt = None
      
      for cnt in contours :
        area = cv2.contourArea(cnt)
        if(max < area) :
          max = area
          maxcnt = cnt

      if maxcnt != np.array([]):
        mask = np.zeros(img.shape).astype(img.dtype)

        # 경계선 내부 255로 채우기
        color = [255, 255, 255]
        cv2.fillPoly(mask, [maxcnt], color)
        img_hand = cv2.bitwise_and(img, mask)
        img_hand = cv2.resize(img_hand, (100,100))
        cv2.imwrite('image.png', img_hand)
        x.append(img_hand)
        y.append(i)
      else:
        continue
  #print(x.shape)
  #print(y.shape)
  return np.array(x),np.array(y)

```
### 얼굴 인식 모델
- 필터를 사용하여 얼굴 색을 추출한뒤 모델학습
- 모델은 resnet50을 이용하여 전이학습
```python
# 모델 생성
def make_model() :
    resnet50 = applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(100,100,3))
    resnet50.trainable = False
    model = models.Sequential()
    model.add(resnet50)
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    #model.add(layers.Dense(256, activation="relu"))

    # 분류 데이터가 몇종류인지
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss = "sparse_categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])
    print(model.summary())
    return model

    # 모델 학습
def learning() :
    train_path = "./data_image"
    test_path = "./test"
    (x_train, y_train)= process_data(train_path)
    (x_test, y_test) = process_non_data(test_path)
    model = make_model()
    # print("x_train : ",x_train[0])
    # print("x_train.shape : ",x_train[0].shape)
    log = model.fit(x_train, y_train, epochs=2, batch_size=16)
    #model.save("face.h5")
    
    model.evaluate(x_test,y_test)
```

### 얼굴 추적 드론
- 학습된 모델로 얼굴을 예측하여 경계값을 찾아내고 사각형을 영상위에 그려냄
- 그린 사각형의 중앙점과 드론캠의 중앙점을 찾아내 두 중앙의 점을 일치 시킴
```python
import cv2
import numpy as np
from djitellopy import Tello

from tensorflow.keras import models
#  -- 초기값 설정 --
# 스타트flag
start_counter = 0
# 프레임 flag - 이때 딥러닝 예측도 실행해야됨
frame_flag = 70
# 임계값 조절
tolerance_x = 5
tolerance_y = 5
# 속도 한계값
slowdown_threshold_x = 13
slowdown_threshold_y = 13
# 속도
drone_speen_x = 13
drone_speen_y = 13
# 화면 조절
set_point_x = 960/2
set_point_y = 720/2
hull = 0
# x,y,w,h 전역변수
# global x, y, w, h
x =0
y =0
w =0
h =0
accuracy_num = ''
img_hand = 0

# 드론 초기화
drone = Tello()
drone.connect()

# # 스트림 연결 오류 
drone.streamon()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1600)

# 모델 임포트
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = models.load_model("last_model.h5")
#x,y,w,h = 0
# ------------ 영상 루프 시작 ------------
while True :
  # 프레임 읽어오기
  frame = drone.get_frame_read().frame
  # takeoff 설정
  if start_counter == 0 :
    drone.takeoff()
    # drone.move_up(80)
    print('takeoff')
    start_counter = 1

  # _, frame = cap.read()

  cv2.circle(frame, (int(set_point_x), int(set_point_y)), 12, (255,0,0), 1) # 화면 중간에 원표시
  cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3 )
  # cv2.putText(frame,f'accuracy = ',accuracy_num,(x+5,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 0.5, cv2.LINE_AA)
  # flalg 설정
  frame_flag -=1
  if frame_flag < 0:
    frame_flag =70
  #print(frame_flag)
  # print("frame_flag", frame_flag)
  if frame_flag == 0 :
    # ------------ 데이터 가공 ------------
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.fastNlMeansDenoisingColored(img_hsv,None,10,10,7,21)
    lower = np.array([0,48,80], dtype="uint8")
    upper = np.array([20,255,255], dtype="uint8")
    img_hand = cv2.inRange(img_hsv,lower,upper)
    
    #경계선 찾음
    contours, _ = cv2.findContours(img_hand, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 영역 찾기
    max = 0
    maxcnt = None
    
    # 채워주기
    for cnt in contours :
      area = cv2.contourArea(cnt)
      if(max < area) :
        max = area
        maxcnt = cnt
    
    # 예외처리
    if maxcnt != np.array([]):
      mask = np.zeros(frame.shape).astype(frame.dtype)
      # 경계선 내부 255로 채우기
      color = [255, 255, 255]
      cv2.fillPoly(mask, [maxcnt], color)
      img_hand = cv2.bitwise_and(frame, mask)

      #print("전",img_hand.shape)
      img_hand_r = cv2.resize(img_hand, (100,100))
      #print(img_hand.shape)
    else :
      continue
    # ------------ 예측 실행 ------------
    # img_hand = np.array([img_hand])
    pred = model.predict(np.array([img_hand_r]))
    print("예측 값 ==>  ",pred)
    # accuracy_num = str(pred[0][1])
    print(accuracy_num)
    pred = np.argmax(pred)
    print("1 : 얼굴 인식,  0 : 인식 실패 ==> ",pred)
    
    if pred == 1 :
      hull = cv2.convexHull(maxcnt)
      x, y, w, h = cv2.boundingRect(hull)
      print("x,y,w,h : ", x,y,w,h)
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3 )
      cv2.circle(frame, (int(x+w/2) , int(y+h/2)), 12, (0,255,0), 1) # 얼굴 중심 표시
      
      frame_flag = 100

      # 도출된 값으로 사각형 그리기
      

      # ------------ 드론 제어 ------------
      #  얼굴중심과 화면중심의 차를 계산
      distance_x = x+w/2 - set_point_x
      distance_y = y+h/2 - set_point_y

      up_down_velocity = 0
      right_left_veiocity = 0
      for_back_veiocity = 0

    # 드론 좌우 이동
      if distance_x < -tolerance_x:
        print("left move")
        right_left_veiocity = - drone_speen_x
      elif distance_x > tolerance_x:
        print("right move")
        right_left_veiocity = drone_speen_x
      else :
        print("OK")

      # 드론 상하 이동
      if distance_y < -tolerance_y:
        print("up move")
        up_down_velocity = drone_speen_y
      elif distance_y > tolerance_y:
        print("down move")
        up_down_velocity = - drone_speen_y
      else :
        print("OK")

      # 드론 앞뒤 이동 및 프레임 크면 넘기기
      # if w*h < 960*720/2:
      #   for_back_veiocity = 10
      # elif w*h > 960*720/2:
      #   for_back_veiocity = -10
      # elif w*h > 960*720 :
      #   for_back_veiocity = 10
      # elif w*h >= 950*700 :
      #   continue
      # else:
      #   print("OK")

      #  임계치 이상 벗어나면 속도 조정 
      if abs(distance_x) < slowdown_threshold_x:
        right_left_veiocity = int(right_left_veiocity / 2)
      if abs(distance_y) < slowdown_threshold_y:
        up_down_velocity = int(up_down_velocity / 2)

      #드론 움직이기
      drone.send_rc_control(right_left_veiocity, 0, up_down_velocity, 0)
    else :
      drone.send_rc_control(0, 0, 0, 0)
      print("0일때 물체 추적")
      continue


  # 비디오 띄우기
  cv2.imshow("Video", frame)
  # cv2.imshow("Video", img_hand)
  # 키 설정
  key = cv2.waitKey(1)
  if key == ord('q'):
    break


drone.streamoff()
cv2.destroyAllWindows()
drone.end()
```

### Day 1 ~ 5
- numpy, matplotlib 기초
- ANN, DNN, CNN 기초
- Tello 드론 제어

## 개발 환경
### 아나콘다
- 파이썬 컴파일러, 라이브러리, 주피터를 이용할 할 수 있는 가상환경
1. 설치
- http://anaconda.com/products/individual

### 아나콘다 기본 명령어
아나콘다의 버전 확인
```bash
conda --version
```
가상환경 생성하기
```bash
conda create -n test python=3.8
```
가상환경 지우기
```bash
conda remove -n test --all
```
생성된 가상환경 확인하기
```bash
conda env list
base                  *  /Users/solstice/opt/anaconda3
test                     /Users/solstice/opt/anaconda3/envs/test
```
가상환경 실행시키기
```bash
conda activate test
```

### 라이브러리
1. conda Prompt 실행 
  - 윈도우는 Anaconda 폴더에 Prompt 존재
  - MacOS에서는 커멘드창에서 가능


2. 가상환경 생성
  - opencv 설치를 위해 3.8로 파이썬 설치
  - jeju 라는 이름의 가상환경 설치
```bash
conda create -n jeju python=3.8
```

3. 가상환경 실행
```bash
conda activate jeju
```
4. 필요 라이브러리 설치
```bash
conda install opencv 
conda install tensorflow   
conda install tellopy
conda install numpy
conda install matplotlib
```
## 딥러닝
### ANN
퍼셉트론 알고리즘
- input 학습데이터
- 모든 W들과 바이어스 b를 0이나 작은 난수로 초기화
- while → 가중치 변경되지 않을때까지 반복
    - 각 데이터의 $x^k$ 와 d^k 에 대하여
    - $y^k(t) = f(w*x)$
    - if  $d^k == y^k(t)$ → continue
    - else  → 가중치 $w_i$에 대하여 $w_i(t+1) = w_i(t) +r($$d^k - y^k(t))x_i^k$


###  DNN
다층 퍼셉트론 - XOR
- 선을 그으면 XOR은 선을 그을 수 없음
- 즉 곡선을 그려 XOR을 풀수 있기 때문에 우리가 선형으로 구할 수 없음
- 아래와 같이 구하는 것을 은닉층이 여러개라고 말하여 DNN이라고함

    | x | y |
    | --- | --- |
    | 0 | 0 |
    | 0 | 1 |
    | 1 | 0 |
    | 1 | 1 |

- NAND 와 OR로 각각 연산 후

    | NAND | OR |
    | --- | --- |
    | 1 | 0 |
    | 0 | 1 |
    | 0 | 1 |
    | 0 | 1 |

-  두 연산을 AND로 연산하면 x,y를 XOR 한 연산 결과 도출

    | AND |
    | --- |
    | 1 |
    | 0 |
    | 0 |
    | 1 |

텐서플로우 tensorflow
```python
  # 라이브러리 임포트
  import tensorflow as tf
  import numpy as np
  from tensorflow.keras import models, layers, datasets

  # 데이터 가공
  (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
  x_train = x_train.reshape(-1, 28 * 28) # 1차원 데이터로 전환
  x_test = x_test.reshape(-1,28 * 28)
  return (x_train,y_train),(x_test,y_test)

  # 모델 설계
  model = models.Sequential()
  model.add(layers.Dense(1000,activation = "relu",input_shape = (28 * 28, ))) # 입력층 1000개
  model.add(layers.Dense(500,activation = "relu")) # 은닉층 500개
  model.add(layers.Dense(10,activation = "softmax")) # 출력층 10개
  
  # 모델 컴파일
  model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
  print(model.summary()) # 모델 요약 출력

  # 로그 분석 
  log = model.fit(x_train, y_train, epochs=3, batch_size=16) 
  lossH = log.history["loss"]
  accH = log.history["accuracy"]
  print(f'loss : {lossH} accuracy:  {accH}')

  # 모델 저장
  model.save("sumin.h5")  

  # 모델 평가
  score = model.evaluate(x_test, y_test) 
  print("모델 평가 점수: ", score)
```

### CNN
Convolutional Neural Networks
- 딥러닝에서 주로 이미지나 영상 데이터 처리할 때 쓰이는 neural network 모델
- DNN은 1차원 형태의 대이터를 사용하는데 만약 이미지 데이터를 사용하면 flatten 시켜 데이터 양상을 변경해야함 이 과정에서 이미지의 손실이 불가피하면서 CNN 탄생
- 특성의 계층을 빌드업하며 이미지 전체보다 이미지의 한 픽섹과 주변 필셀들의 연관성을 살림
- 즉 사진의 전반적인 부분이 아닌 한 주요한 한 특징만을 보는 것

Convolution
- MNIST Dataset에서 가져온 이미지는 gray scale 이미지로 28*28 행렬로 표현
- 2차원 이미지를 matrix로 표현 가능
- input image가 $5*5$이고 필터(커널)이 $3*3$ 인 경우 이미지와 필터를 `inner product` 연산 진행 

Zero Padding
- 컨볼루션 처리시 이미지 데이터 손실이 발생함
- `Padding` 방식으로 해결 가능
- 0으로 구성된 테두리를 이미지 가장자리에 감싸 줌

## Tello 드론
- 파이썬을 이용하여 드론 프로그래밍 가능한 드론
- 텔로 드론은 무선 연결 통신
### 드론 제어 기초
1. 와이파이 연결
- 드론의 전원 on
- tello 드론과 연결할 제어기기에 드론의 와이파이로 연결
2. 임포트 및 드론 연결
```python
# 임포트
from djitellopy import Tello

# 텔로 드론 객체 생성
tello = Tello()

# 드론 연결
tello.connect()
# 이륙
tello.takeoff()
# 왼쪽으로 100cm 이동
tello.move_left(100)
# 앞쪽으로 100cm 이동
tello.move_forward(100)
# 착륙
tello.land()
# 연결끊기
tello.end()
```
3. 실시간 스트리밍
```python
tello.streamon()
tello.streamoff()
```
