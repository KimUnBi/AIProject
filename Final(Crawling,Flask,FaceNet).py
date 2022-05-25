#!/usr/bin/env python
# coding: utf-8

# # 필요한 lib 설치 

# In[5]:


pip install tensorflow
pip install MTCNN
pip install selenium


# # 모델로딩 

# In[2]:


from keras.models import load_model
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import mtcnn
from PIL import Image
from os import listdir #폴더 읽어오기
import os
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
#모델 적합화 시키기(SVM활용하기) --> sklearn SVC클래스
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from random import choice
from random import choice


# # 이미지 로딩 및 얼굴 추출 함수 생성 

# In[77]:


# 지금까지 과정 하나로 묶어서 함수 만들기
# 주어진 사진에서 하나의 얼굴 추출하는 함수
def extract_face(filename, required_size=(160, 160)):
	# 파일에서 이미지 불러오기
	image = Image.open(filename)
	# RGB로 변환, 필요시
	image = image.convert('RGB')
	# 배열로 변환
	pixels = np.asarray(image)
	# 감지기 생성, 기본 가중치 이용
	detector = mtcnn.MTCNN()
	# 이미지에서 얼굴 감지
	results = detector.detect_faces(pixels)
	if len(results) == 0 :
		return 0
	else:
		# 첫 번째 얼굴에서 경계 상자 추출 ----> ★★★여러개 추출할 수 있도록 변동 필요함!!!!!!!★★★
		x1, y1, width, height = results[0]['box']
	# 버그 수정
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
	# 얼굴 추출
		face = pixels[y1:y2, x1:x2]
	# 모델 사이즈로 픽셀 재조정
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = np.asarray(image)
		return face_array

#디렉토리 안의 모든 이미지를 불러오고 이미지에서 얼굴 추출
def load_faces(directory):
	faces = list()
	# 파일 열거
	for filename in listdir(directory):
		# 경로
		path = directory + filename
		# 얼굴 추출
		face = extract_face(path)
		# 저장
		faces.append(face)
	return faces

# 이미지를 포함하는 각 클래스에 대해 하나의 하위 디렉토리가 포함된 데이터셋을 불러오기
def load_dataset(directory):
	X, y = list(), list()
	# 클래스별로 폴더 열거
	for subdir in listdir(directory):
		# 경로
		path = directory + subdir + '/'
		# 디렉토리에 있을 수 있는 파일을 건너뛰기(디렉토리가 아닌 파일)
		if not os.path.isdir(path):
			continue
		# 하위 디렉토리의 모든 얼굴 불러오기
		faces = load_faces(path)
		# 레이블 생성
		labels = [subdir for _ in range(len(faces))]
		# 진행 상황 요약
		print('>%d개의 예제를 불러왔습니다. 클래스명: %s' % (len(faces), subdir))
		# 저장
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)


# In[78]:


trainX, trainy=load_dataset('archive/train/')


# In[79]:


testX, testy = load_dataset('archive/val/')


# In[81]:


print(trainX.shape)
print(trainy.shape)
print(testX.shape[0])
print(testy.shape)


# In[ ]:


trainX, trainy = load_dataset('archive/train/')
# 테스트 데이터셋 불러오기
testX, testy = load_dataset('archive/val/')
# 배열을 단일 압축 포맷 파일로 저장
np.savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)


# # 학습된 FaceNet활용하여 얼굴 임베딩 값 얻어오기 

# In[45]:


# 하나의 얼굴의 얼굴 임베딩 얻기
def get_embedding(model, face_pixels):
	# 픽셀 값의 척도
	face_pixels = face_pixels.astype('int32')
	# 채널 간 픽셀값 표준화(전역에 걸쳐)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# 얼굴을 하나의 샘플로 변환
	samples = np.expand_dims(face_pixels, axis=0)
	# 임베딩을 갖기 위한 예측 생성
	yhat = model.predict(samples)
	return yhat[0]

def embedding_final():
    # 얼굴 데이터셋 불러오기
    data = np.load('faces-dataset.npz', allow_pickle=True)
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('불러오기: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    # facenet 모델 불러오기
    model =  load_model('facenet_keras.h5')
    # 훈련 셋에서 각 얼굴을 임베딩으로 변환하기
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)
    print(newTrainX.shape)
    # 테스트 셋에서 각 얼굴을 임베딩으로 변환하기
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = np.asarray(newTestX)
    # 배열을 하나의 압축 포맷 파일로 저장
    np.savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)


# # 안면 이미지 유사도 분석 확인

# In[ ]:


# 데이터셋 불러오기
data = np.load('faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('데이터셋: 훈련=%d, 테스트=%d' % (trainX.shape[0], testX.shape[0]))
# 데이터 모델링하기전 얼굴 임베딩 vector normalizer 진행하기(vector의 길이가 1이나 단위길이가 될때 값을 스케일링하기)
# sklearn의 Normalizer 클래스 사용하기
# 입력vector 일반화
# 입력 벡터 일반화
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# 목표 레이블 암호화(이름 변수 문자열 정수로 변환하기)
out_encoder = preprocessing.LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)


model = SVC(kernel='linear')
model.fit(trainX, trainy)

#모델 평가하기(분류 정확도 계산하기)
# 예측
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# 정확도 점수
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# 요약
print('정확도: 훈련=%.3f, 테스트=%.3f' % (score_train*100, score_test*100))


# # 확인해보기 

# In[8]:


def final_model() :
    # 얼굴 불러오기
    data = np.load('faces-dataset.npz')
    testX_faces = data['arr_0']

    # 얼굴 임베딩 불러오기
    data = np.load('faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    # 입력 벡터 일반화
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    # 목표 레이블 암호화
    out_encoder = preprocessing.LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # 모델 적합
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    # 테스트 데이터셋에서 임의의 예제에 대한 테스트 모델
    #selection = choice([i for i in range(testX.shape[0])])
    selection = choice([i for i in range(testX.shape[0])])
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    random_face_class = testy[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])

    # 얼굴 예측
    samples = np.expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)

    # 이름 얻기
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('예상: %s (%.3f)' % (predict_names[0], class_probability))
    print('추측: %s' % random_face_name[0])

    # 재미삼아 그리기
    plt.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    plt.title(title)
    plt.show()


# # 크롤링 코드 

# In[12]:


from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus
from selenium import webdriver
import json
import os
import urllib.request
from urllib.parse import quote_plus
import time


# In[55]:


def crawling_people():
    #찾고자 하는 검색어 입력
    index = 0
    for page_num in range(1,20) :
        url = 'https://www.ppomppu.co.kr/zboard/zboard.php?id=free_gallery&page={}&category=3&divpage=61'.format(str(page_num))
        img_url = []

        browser = webdriver.Chrome()
        browser.get(url)

        # User-Agent를 통해 봇이 아닌 유저정보라는 것을 위해 사용
        header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        # 소스코드가 있는 경로에 '검색어' 폴더가 없으면 만들어준다.(이미지 저장 폴더를 위해서) 
        # if not os.path.exists(searchterm):
        #     os.mkdir(searchterm)
        a_tag = browser.find_elements_by_css_selector('.han a')
        # 가로 = 0, 세로 = 10000 픽셀 스크롤한다.
        browser.execute_script("window.scrollBy(0,10000)")
        #._image._listImage
        imgs = browser.find_elements_by_css_selector('.gallery_img img')
        time.sleep(5)
        for i in imgs:
            temp = i.get_attribute('src')
            img_url.append(temp)

        img_folder = 'archive/val/img'
        if not os.path.isdir(img_folder):
            os.mkdir(img_folder)

        for link in img_url:
            index += 1
            urllib.request.urlretrieve(link, f'archive/val/img/{index}.jpg')


# In[56]:


crawling_people()


# # Flask 연동 

# In[6]:


pip install flask
pip install werkzeug==2.0.3
pip install flask_cors


# In[6]:


from flask import Flask
from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS


# In[84]:


app = Flask(__name__)
CORS(app)
@app.route("/")
def start():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    trainX, trainy = load_dataset('archive/train/')
    # 테스트 데이터셋 불러오기
    testX, testy = load_dataset('archive/val/')
    # 배열을 단일 압축 포맷 파일로 저장
    np.savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)
    embedding_final()
    #모델 사용해보기
    final_model()
    return "값"

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port ="8090")


# In[ ]:




