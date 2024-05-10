#!/bin/bash

# Install Tesseract OCR
apt-get update
apt-get install -y tesseract-ocr

# Install English language data
apt-get install -y tesseract-ocr-eng
