#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Tesseract OCR
apt-get install -y tesseract-ocr
apt-get install -y libtesseract-dev

# Install English language data
apt-get install -y tesseract-ocr-eng
