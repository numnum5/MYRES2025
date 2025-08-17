# Multiple Choice Exam Processing Tool

## Overview
Takes a PDF scan of the exam sheet which can be 1 to 2 pages long depending on the number of questions being marked.
Prints out the answers in a list format

## Prerequsites
Windows:
1. Install Python

Make sure to check “Add Python to PATH” during installation.

3. Install Tesseract-OCR

Download the Tesseract Windows installer.

Install it (e.g., into C:\Program Files\Tesseract-OCR).

Add the Tesseract installation folder (e.g., C:\Program Files\Tesseract-OCR) to your system PATH.

Linux: 
1. Install Python

sudo apt update

sudo apt install python3 python3-pip

2. Install Tesseract-OCR

sudo apt update

sudo apt install tesseract-ocr

3. Install Poppler

sudo apt update

sudo apt install poppler-utils

## Installation
1. Clone this repository:
git clone https://github.com/numnum5/MYRES2025.git
2. cd MYRES2025
3. Install dependencies:
pip install -r requirements.txt

## Usage
python prototype.py <pdf_file> [number_of_questions]
