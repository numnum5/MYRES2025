import os
import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path

# Class encapsulating logic for converting PDF files to image png formats
class Converter:
    def __init__(self, filepath, output_dir="template", dpi=300):
        # Make paths relative to script location
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.join(self.base_dir, filepath)
        self.output_dir = os.path.join(self.base_dir, output_dir)
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)


    # Checks if a PDF has a single page
    def is_single_page(self):
        info = pdfinfo_from_path(self.filepath)
        return info["Pages"]

    def convert_pages(self):
        number_of_pages = self.is_single_page()
        if self.is_single_page() == 1:
            return self.convert_single_page(page_number=1), 1
        else:
            return (self.convert_all_pages(), number_of_pages)


    # Converts PDF scans with multiple pages
    def convert_all_pages(self):
        images = convert_from_path(self.filepath, dpi=self.dpi)
        output_paths = []
        for i, image in enumerate(images, start=1):
            output_path = os.path.join(self.output_dir, f"page{i}.png")
            image.save(output_path, "PNG")
            output_paths.append(output_path)
        return output_paths
        


    # Convert single page pdf
    def convert_single_page(self, page_number=1):
        images = convert_from_path(
            self.filepath,
            dpi=self.dpi,
            first_page=page_number,
            last_page=page_number
        )
        if not images:
            raise ValueError(f"Page {page_number} not found.")
        output_path = os.path.join(self.output_dir, f"page{page_number}.png")
        images[0].save(output_path, "PNG")
        return output_path
    


    # Aligns images to template to remove tilt and distortion of PDF scans
    # Reference: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    def align_to_template(self, input_image_path, output_path, page_number, max_features=500, good_match_percent=0.15):
        template_path = os.path.join(self.base_dir, 'template', 'page1.png')
        if(page_number > 1):
            print("AIJDOAJDOIAJD")
            template_path = os.path.join(self. base_dir, 'template', 'page2.png')
        img1 = cv2.imread(template_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        # # Read image to be aligned
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors
        # MAX_NUM_FEATURES = 500
        orb = cv2.ORB_create(max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

        # Converting to list for sorting as tuples are immutable objects.
        matches = list(matcher.match(descriptors1, descriptors2, None))

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * good_match_percent)
        matches = matches[:numGoodMatches]

        im_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype = np.float32)
        points2 = np.zeros((len(matches), 2), dtype = np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)


        height, width, _ = img1.shape
        img2_reg = cv2.warpPerspective(img2, h, (width, height))
        cv2.imwrite(output_path, img2_reg)
        return output_path
