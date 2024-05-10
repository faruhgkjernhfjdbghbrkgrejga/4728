import streamlit as st
from google.cloud import vision
from googletrans import Translator
import io
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageEnhance

# 시크릿 로드
key_path = st.secrets["google"]["key_path"]

# Google Cloud Vision API 클라이언트 설정
client = vision.ImageAnnotatorClient.from_service_account_file(key_path)

# Translator 객체 생성
translator = Translator()

def extract_text_from_image(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

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
    extracted_text_1 = extract_text_from_image(io.BytesIO(uploaded_file.read()))

    # 번역 및 이미지 처리 코드 ...

    # 처리된 이미지 표시
    st.image(image, caption="번역된 이미지", use_column_width=True)
