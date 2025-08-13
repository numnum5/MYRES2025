import os
import argparse
from converter import Converter
from extractor import Extractor
from predictor import Predictor
from detailExtractor import DetailExtractor
from detailPredictor import DetailPredictor
def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Process a PDF and predict answers.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("questions", type=int, help="Number of questions in the PDF")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    number_of_questions = args.questions

    # Validate PDF file
    if not os.path.isfile(pdf_path):
        parser.error(f"The file {pdf_path} does not exist.")

    # Prepare directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "pages")
    os.makedirs(output_dir, exist_ok=True)

    # Conversion
    converter = Converter(pdf_path, output_dir)
    output_paths, number_of_pages = converter.convert_pages()

    # Alignment
    if number_of_pages > 1:
        for page_number, output_path in enumerate(output_paths):
            if page_number + 1 > 1: 
                converter.align_to_template(output_path, output_path, page_number + 1, 800)
            else:
                converter.align_to_template(output_path, output_path, page_number + 1, good_match_percent=0.12)
    else:
        converter.align_to_template(output_paths, output_paths, 1)




    # Dictionary to store final output
    answers = {
        
    }

    # Extract student details from the first page
    detailExtractor = DetailExtractor(os.path.join(output_dir, "page1.png"))
    detailExtractor.extract()
    detailPredictor = DetailPredictor(os.path.join(base_dir, 'student_number'))
    answers['student_number'] = detailPredictor.predict_student_number()
    detailPredictor = DetailPredictor(os.path.join(base_dir, 'initials'))
    answers['initials'] = detailPredictor.predict_initials()
    detailPredictor = DetailPredictor(os.path.join(base_dir, 'surname'))
    answers['surname'] = detailPredictor.predict_initials()
    

    # Extraction
    extractor = Extractor(output_dir, number_of_pages)
    questions_dir = extractor.extract()

    # Prediction
    predictor = Predictor(questions_dir, number_of_questions)
    answers['questions'] = predictor.predict()

    print("Predicted Answers:", answers)

if __name__ == "__main__":
    main()

