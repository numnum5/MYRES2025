# Multiple Choice Exam Processing Tool

## Overview
Takes a PDF scan of the exam sheet which can be 1 to 2 pages long depending on the number of questions being marked.
Prints out the answers in a list format

## Prerequsites
1. Install Python
\nsudo apt update
\nsudo apt install python3 python3-pip
2. Install Tesseract-OCR
\nsudo apt update
\nsudo apt install tesseract-ocr

## Installation
1. Clone this repository:
git clone https://github.com/numnum5/MYRES2025.git
2. cd MYRES2025
3. Install dependencies:
pip install -r requirements.txt

## Usage
python prototype.py <pdf_file> [number_of_questions]
