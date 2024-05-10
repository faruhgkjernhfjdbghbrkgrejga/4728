import streamlit as st
from google.cloud import vision
from googletrans import Translator
import io
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageEnhance

# Google Cloud 프로젝트의 서비스 계정 키를 설정합니다.
key_path = r"C:\Users\kimjunghoo\Desktop\pycode\second-hexagon-419606-bb9716b159fc.json"
client = vision.ImageAnnotatorClient.from_service_account_json(key_path)
# Google Cloud 프로젝트의 서비스 계정 키를 설정합니다.
credentials = service_account.Credentials.from_service_account_file('service_account_key.json')
client = vision.ImageAnnotatorClient(credentials=credentials)


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

# 이미지 파일 경로 설정
image_path = r'C:\Users\kimjunghoo\Desktop\pycode\example1.jpg'

# 이미지에서 텍스트 추출
extracted_text_1 = extract_text_from_image(image_path)

# 번역된 텍스트를 저장할 리스트
translated_texts = []

# 번역할 언어 설정 (예: 한국어로 번역)
target_language = 'ko'

for text_info in extracted_text_1:
    text = text_info['text']
    print(text)
    # 텍스트 언어 감지
    detected_language = translator.detect(text).lang
    # 번역 수행
    translated_text = translator.translate(text, src=detected_language, dest=target_language)
    translated_texts.append(translated_text.text)  # 번역된 텍스트를 저장
    print(translated_text.text)
    print()

#translated_texts의 전문 해석본과 token해석본을 비교 및 리스트 업데이트##########
ful_trans = translated_texts[0].split()
print(ful_trans)
print(len(ful_trans))
dif_list = []
for i in range(len(ful_trans)):
        if ful_trans[i] != translated_texts[i+1]:
            print(f"두 리스트의 {i}번째 요소가 다릅니다: {ful_trans[i]} != {translated_texts[i+1]}")
            dif_list.append(i)
i=0
while(i <= len(ful_trans)):
     if i in dif_list:
          j=i+1
          while(j in dif_list):
               j+=1 #i부터 j까지 다름
          j-=1
          index=i+1
          translated_texts[index] = ful_trans[i]
          while(i+1<=j):
               i+=1
               translated_texts[index] =translated_texts[index] + ful_trans[i]
               translated_texts[i+1] = "null"
     i=i+1
print(translated_texts)
#######################

# 이미지 열기
image = Image.open(image_path)

# 이미지 배열로 변환
draw = ImageDraw.Draw(image)

# 추출된 텍스트를 이미지 위에 쓰기
for i, text_info in enumerate(extracted_text_1):
    coordinates = text_info['coordinates']
    draw.rectangle([(coordinates[0][0], coordinates[0][1]), (coordinates[2][0], coordinates[2][1])], fill=(255, 255, 255, 100))  # 사각형 투명도 조절

# 이미지 저장
image.save("image_with_text_and_rectangles.jpg")

 # 텍스트가 있는 위치에 사각형 그리기
for i, text_info in enumerate(extracted_text_1):
        coordinates = text_info['coordinates']
        draw.rectangle([(coordinates[0][0], coordinates[0][1]), (coordinates[2][0], coordinates[2][1])], fill="white")

        ##

for i, text_info in enumerate(extracted_text_1):
    if len(translated_texts) <= i+1:
         break
    if i != 0:  # 첫 번째 텍스트는 이미지에서 지우지 않음 #첫 번째 텍스트는 전문이 옴
        coordinates = text_info['coordinates']
        # 번역된 텍스트를 이미지 위에 쓰기
        #translated_text_print = translated_texts[i]  # 첫 번째 번역된 텍스트만 사용
        text_position = (coordinates[0][0], coordinates[0][1])  # 텍스트 상자의 왼쪽 상단 좌표를 사용합니다.
        box_height = coordinates[3][1] - coordinates[0][1]  # 상자의 높이 계산
        # 적절한 폰트 크기 설정
        font_size = min(int(box_height), 36)  # 상자의 높이 또는 최대 36px로 폰트 크기 설정
        font = ImageFont.truetype(r"C:\Users\kimjunghoo\Desktop\pycode\nanum-all\나눔 글꼴\나눔고딕\NanumFontSetup_TTF_GOTHIC\NanumGothic.ttf", font_size , encoding="utf-8")  # 폰트와 크기 설정
        draw.text(text_position, translated_texts[i], fill='black', font=font)

# 수정된 이미지 저장
image.save("image_with_text.jpg")

st.title("이미지 번역 앱")

# 파일 업로드 위젯
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

