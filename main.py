import streamlit as st
from google.cloud import vision
from googletrans import Translator
import io
from PIL import Image, ImageDraw, ImageFont
import os

# 시크릿에서 서비스 계정 키 파일 내용 가져오기
service_account_info = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# 서비스 계정 키 파일 내용을 환경 변수에 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_info

# Google Cloud Vision API 클라이언트 설정
client = vision.ImageAnnotatorClient.from_service_account_info(service_account_info)

# Translator 객체 생성
translator = Translator()

def extract_text_from_image(image_bytes):
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    extracted_text = []
    for text in texts:
        text_info = {}
        text_info['text'] = text.description
        text_info['coordinates'] = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        extracted_text.append(text_info)

    return extracted_text

# 이미지 파일 업로드
uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 파일을 열기
    image = Image.open(io.BytesIO(uploaded_file.read()))

    # 이미지 표시
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # 이미지에서 텍스트 추출 및 번역
    extracted_text_1 = extract_text_from_image(uploaded_file.read())

    # 번역할 언어 선택
    target_lang = st.selectbox("번역할 언어를 선택하세요", ["en", "ko", "ja", "zh-cn"])

    # 번역된 텍스트 저장할 리스트
    translated_texts = []

    # 텍스트 번역
    for text_info in extracted_text_1:
        text = text_info['text']
        translation = translator.translate(text, dest=target_lang)
        translated_text = translation.text
        translated_texts.append(translated_text)

     # 번역된 텍스트를 이미지에 렌더링
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 36)
    for i, text_info in enumerate(extracted_text_1):
        coordinates = text_info['coordinates']
        draw.rectangle([(coordinates[0][0], coordinates[0][1]), (coordinates[2][0], coordinates[2][1])], fill="white")
        text_position = (coordinates[0][0], coordinates[0][1])
        draw.text(text_position, translated_texts[i], font=font, fill='black')

    # 처리된 이미지 표시
    st.image(image, caption="번역된 이미지", use_column_width=True)

