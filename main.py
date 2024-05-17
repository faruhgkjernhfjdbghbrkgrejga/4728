import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import torch.nn as nn
import pytesseract
from google.cloud import translate_v2 as translate

# UNet 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 인코더
        self.down1 = self.contract_block(3, 64, 3, 1)
        self.down2 = self.contract_block(64, 128, 3, 1)
        self.down3 = self.contract_block(128, 256, 3, 1)
        self.down4 = self.contract_block(256, 512, 3, 1)

        # 최소 크기에 도달
        self.middle = self.contract_block(512, 1024, 3, 1)

        # 디코더
        self.up4 = self.expand_block(1024, 512, 3, 1)
        self.up3 = self.expand_block(512, 256, 3, 1)
        self.up2 = self.expand_block(256, 128, 3, 1)
        self.up1 = self.expand_block(128, 64, 3, 1)

        # 최종 컨볼루션
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def __call__(self, x):
        # 인코더
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # 중간
        m = self.middle(d4)

        # 디코더
        up4 = self.up4(m)
        up3 = self.up3(up4 + d4)
        up2 = self.up2(up3 + d3)
        up1 = self.up1(up2 + d2)

        # 최종 컨볼루션
        out = self.final(up1)
        return out

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        return expand

def transform_image(image):
    # 이미지 전처리
    image = TF.to_tensor(image)
    image = TF.resize(image, size=(256, 256))
    return image.unsqueeze(0)  # 배치 차원 추가

# Google Cloud Translation API 클라이언트 설정
translate_client = translate.Client()

def ocr_image(image_path):
    """ 이미지에서 텍스트 추출 """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def translate_text(text, target_language='en'):
    """ 텍스트 번역 """
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# 스트림릿 앱 시작
st.title('이미지 번역기')

# 파일 업로더
uploaded_file = st.file_uploader("이미지 파일을 선택하세요.", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    # 이미지 변환
    model = UNet()  # 모델 인스턴스 생성
    input_image = transform_image(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_image = input_image.to(device)  # Added this line to move input_image to the same device as the model
    with torch.no_grad():
        output_image = model(input_image)
        output_image = TF.to_pil_image(output_image.squeeze(0))

    st.image(output_image, caption='변환된 이미지', use_column_width=True)
    output_image.save('translated_image.png')

    # OCR을 통해 이미지에서 텍스트 추출
    extracted_text = ocr_image('translated_image.png')
    st.write("추출된 텍스트:", extracted_text)

    # 추출된 텍스트를 영어로 번역
    translated_text = translate_text(extracted_text, 'en')
    st.write("번역된 텍스트:", translated_text)
