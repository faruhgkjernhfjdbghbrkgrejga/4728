import streamlit as st
from google.oauth2 import service_account
from google.cloud import vision
from google.cloud import translate_v2 as translate
from PIL import Image, ImageDraw, ImageFont
import io
import os

# 서비스 계정 키 정보를 직접 사용
credentials = service_account.Credentials.from_service_account_info({
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"].replace('\\n', '\n'),
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": st.secrets["auth_uri"],
    "token_uri": st.secrets["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["client_x509_cert_url"]
})

vision_client = vision.ImageAnnotatorClient(credentials=credentials)
translate_client = translate.Client(credentials=credentials)

def extract_text(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def translate_text(text, target_language="ko"):
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("tempDir", uploaded_file.name)
    except Exception as e:
        st.error("파일 저장 중 오류가 발생했습니다.")
        return None

def draw_text_on_image(image_path, text):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    # 기본 폰트 사용
    font = ImageFont.load_default()
    text_width, text_height = draw.textsize(text, font=font)
    width, height = image.size
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    draw.text((x, y), text, font=font, fill="red")
    # 이미지를 바이트 스트림으로 변환하여 반환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

st.title('이미지 번역기')

uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image_path = save_uploaded_file(uploaded_file)
    if image_path:
        image = Image.open(image_path)
        st.image(image, caption='업로드된 이미지', use_column_width=True)
        extracted_text = extract_text(image_path)
        if extracted_text:
            st.write("추출된 텍스트:", extracted_text)
            translated_text = translate_text(extracted_text, 'ko')
            st.write("번역된 텍스트:", translated_text)
            image_with_text = draw_text_on_image(image_path, translated_text)
            st.image(image_with_text, caption='번역된 이미지', use_column_width=True, format='PNG')
        else:
            st.write("이미지에서 텍스트를 찾을 수 없습니다.")
            st.write("이미지에서 텍스트를 찾을 수 없습니다.")
