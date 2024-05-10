import streamlit as st
import pytesseract
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

# Tesseract OCR 엔진 경로 지정
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 윈도우 경로 예시

# Translator 객체 생성
translator = Translator()

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# 이미지 파일 업로드
uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 파일을 열기
    image = Image.open(uploaded_file)

    # 이미지 표시
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # 이미지에서 텍스트 추출
    extracted_text = extract_text_from_image(image)

    # 번역할 언어 선택
    target_lang = st.selectbox("번역할 언어를 선택하세요", ["en", "ko", "ja", "zh-cn"])

    # 텍스트 번역
    translation = translator.translate(extracted_text, dest=target_lang)
    translated_text = translation.text

    # 번역된 텍스트 렌더링
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 36)
    draw.text((10, 10), translated_text, font=font, fill='red')

    # 처리된 이미지 표시
    st.image(image, caption="번역된 이미지", use_column_width=True)
