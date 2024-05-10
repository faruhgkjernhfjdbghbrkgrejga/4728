import streamlit as st
from google.cloud import vision
from google.cloud import translate
from PIL import Image, ImageDraw, ImageFont
import io
import os

# 시크릿에서 서비스 계정 키 파일 경로 가져오기
service_account_path = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# 서비스 계정 키 파일 경로를 환경 변수에 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

# Google Cloud 클라이언트 초기화
vision_client = vision.ImageAnnotatorClient()
translate_client = translate.Client()

# 웹 페이지 제목
st.title("이미지 번역기")

# 이미지 업로드 위젯
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])

if uploaded_file is not None:
    # 이미지 데이터 로드
    image = Image.open(uploaded_file)
    
    # 텍스트 인식
    content = vision_client.text_detection(image=vision_client.encode_image(image))
    texts = content.text_annotations
    
    # 번역할 언어 선택
    target_lang = st.selectbox("번역할 언어를 선택하세요", ["en", "ko", "ja", "zh-cn"])
    
    # 추출된 텍스트와 번역된 텍스트 저장할 리스트
    extracted_texts = []
    translated_texts = []
    
    # 텍스트 추출 및 번역
    for text in texts:
        extracted_text = text.description
        extracted_texts.append(extracted_text)
        
        translation = translate_client.translate(extracted_text, target_language=target_lang)
        translated_text = translation["translatedText"]
        translated_texts.append(translated_text)
    
    # 번역된 텍스트를 이미지에 렌더링
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 36)
    for i, text_info in enumerate(texts):
        coordinates = [(vertex.x, vertex.y) for vertex in text_info.bounding_poly.vertices]
        draw.rectangle([(coordinates[0][0], coordinates[0][1]), (coordinates[2][0], coordinates[2][1])], fill="white")
        text_position = (coordinates[0][0], coordinates[0][1])
        draw.text(text_position, translated_texts[i], font=font, fill='black')
    
    # 결과 이미지 출력
    st.image(image)
