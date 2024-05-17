import streamlit as st
from google.cloud import vision
from google.cloud import translate_v2 as translate
from PIL import Image, ImageDraw, ImageFont
import io
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]


def extract_text(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def translate_text(text, target_language="ko"):
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return result['translatedText']

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("tempDir", uploaded_file.name)
    except Exception as e:
        return None

def draw_text_on_image(image_path, text):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_width, text_height = draw.textsize(text, font=font)
    width, height = image.size
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    draw.text((x, y), text, font=font, fill="red")
    return image

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
            st.image(image_with_text, caption='번역된 이미지', use_column_width=True)
        else:
            st.write("이미지에서 텍스트를 찾을 수 없습니다.")

