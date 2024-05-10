import streamlit as st
from google.cloud import vision
from googletrans import Translator
import io
from PIL import Image, ImageDraw, ImageFont

# Google Cloud 프로젝트의 서비스 계정 키를 설정합니다.
key_path = "service_account_key.json"
client = vision.ImageAnnotatorClient.from_service_account_file(key_path)

# Translator 객체 생성
translator = Translator()

def extract_text_from_image(image):
    content = image.read()

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

st.title("이미지 번역 앱")

# 파일 업로드 위젯
uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 파일을 이미지로 열기
    image = Image.open(io.BytesIO(uploaded_file.read()))

    # 이미지 표시
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # 이미지에서 텍스트 추출 및 번역
    extracted_text = extract_text_from_image(uploaded_file)

    # 번역된 텍스트 출력
    st.write("### 번역된 텍스트:")
    for text_info in extracted_text:
        text = text_info['text']
        st.write("- 원본 텍스트:", text)
        
        # 번역 수행
        translated_text = translator.translate(text, dest='ko')
        st.write("- 번역 결과:", translated_text.text)
