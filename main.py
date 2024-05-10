import streamlit as st
from google.cloud import vision
from google.cloud import translate
from PIL import Image, ImageDraw, ImageFont
import io
import os

# .streamlit/secrets.toml
GOOGLE_APPLICATION_CREDENTIALS = "/path/to/service_account.json"

# 시크릿에서 서비스 계정 키 파일 경로 가져오기
service_account_path = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# 서비스 계정 키 파일 경로를 환경 변수에 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

# Google Cloud 클라이언트 초기화
vision_client = vision.ImageAnnotatorClient()
translate_client = translate.Client()


# 이미지 업로드 위젯
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])

if uploaded_file is not None:
    # 이미지 데이터 로드
    image = Image.open(uploaded_file)
    
    # 텍스트 인식
    content = vision_client.text_detection(image=vision_client.encode_image(image))
    text = content.text
    
    # 번역할 언어 선택
    target_lang = st.selectbox("번역할 언어를 선택하세요", ["en", "ko", "ja", "zh-cn"])
    
    # 텍스트 번역
    translation = translate_client.translate(text, target_language=target_lang)
    translated_text = translation["translatedText"]
    
    # 번역된 텍스트를 이미지에 렌더링
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 36)
    draw.text((10, 10), translated_text, font=font, fill=(0, 0, 0))
    
    # 결과 이미지 출력
    st.image(image)
