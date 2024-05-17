import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from google.cloud import translate_v2 as translate

# Google 번역 클라이언트 설정
translate_client = translate.Client()

def ocr_image(image, src_lang='eng'):
    """ 이미지에서 텍스트 추출 """
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Tesseract 경로 설정
    return pytesseract.image_to_string(image, lang=src_lang)

def translate_text(text, src_lang='en', dest_lang='ko'):
    """ 텍스트 번역 """
    return translate_client.translate(text, source_language=src_lang, target_language=dest_lang)['translatedText']

def draw_text_on_image(image, text):
    """ 이미지에 텍스트 그리기 """
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
src_lang = st.text_input("출발 언어 코드 (예: en, ko)", 'en')
dest_lang = st.text_input("목표 언어 코드 (예: en, ko)", 'ko')

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    extracted_text = ocr_image(image, src_lang)
    st.write("추출된 텍스트:", extracted_text)

    translated_text = translate_text(extracted_text, src_lang, dest_lang)
    st.write("번역된 텍스트:", translated_text)

    image_with_text = draw_text_on_image(image, translated_text)
    st.image(image_with_text, caption='번역된 이미지', use_column_width=True)
