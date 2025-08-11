import cv2
import numpy as np
import os
import re
import pytesseract

# Class encapsulating logic for predictor each question extracted from PDF scans
class Predictor:


    answer_key = {
        0 : 'A',
        1 : 'B',
        2 : 'C',
        3 : 'D',
        4 : 'E'
    }
    
    FILLED_PIXELS_THRESHOLD = 900
    CROSSED_RADIUS_THRESHOLD = 35

    def __init__(self, filepath, numQs):
        self.filepath = filepath
        self.numQs = numQs


    def CheckCrossedBubbles(self, filled : dict) -> dict:
        candidates = []
        for k in filled:
            current = filled[k]
            if current['filled']:
                if current['circle'][2] > self.CROSSED_RADIUS_THRESHOLD:
                    filled[k]['crossed'] = True
                    candidates.append(k)
                else:
                    filled[k]['crossed'] = False
        return filled, candidates

    # Flow:
    # Find filled in bubbles
    # If there any crossed out circles, and if there at least 1, it would be the non crossed out filled in circle
    # If there are two crossed out circles we check the letter on the left region of ROI
    def predictAnswer(self, filled, left_part):
        gray = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
        cropped = gray[5:-5, :]
        blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
        
        # cv2.imshow("Detected Circles (Scaled Display)", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Configure pytesseract to detect characters ABCDE only
        text = pytesseract.image_to_string(blurred, config="--psm 10 -c tessedit_char_whitelist=ABCDE").strip()
        # print(text)
        if text and len(text) == 1 and text in ['A', 'B', 'C', 'D', 'E']:
            # print(text)
            return text


        # If no letter is detected on the left regio of ROI
        # Check if theres more than 1 filled in bubbles
        if len(filled) > 1:

            # Pick the best choice which is the bubble with smallest radius
            best_choice = None
            min_radius = float('inf')
            for k in filled:
                info = filled.get(k, {})
                if not info.get("crossed", False):  # skip if crossed
                    r = info.get("circle", (0, 0, float('inf')))[2]
                    if r < min_radius:
                        min_radius = r
                        best_choice = k

            _, candidates = self.CheckCrossedBubbles(filled)

            # If there are no potential crossed out bubbles
            if len(candidates) == 0:
                for k in filled:
                    # Check if filled in bubbles has a larger enough a difference in size
                    # compared to the best choice (Smallest bubble) to make it a
                    # crossed out circle. Return the best choice if it is
                    # Threshold is set to 6 but can be change
                    if abs(filled[best_choice]['circle'][2] - filled[k]['circle'][2]) >= 6:
                        return filled[best_choice]['key']
                    # Calculate the radius difference between the filled ones
                    # if difference is minimal probably they are filled not crossed out
                
                return ''
            else:
                return filled[best_choice]['key']
        # If only bubble is filled return that
        if len(filled) == 1:
            key = list(filled.keys())[0]
            return filled[key]['key']
        
        # Return an empty string if none of the above conditions meet
        return ''

    def detectFilledInbubbles(self, unique_circles, gray, image):
        filled = {}
        for i, c in enumerate(unique_circles):
            x, y = int(c[0]), int(c[1])
            radius = int(c[2])
            padding = 5 
            top_left_x = max(x - radius - padding, 0)
            top_left_y = max(y - radius - padding, 0)
            bottom_right_x = min(x + radius + padding, image.shape[1])
            bottom_right_y = min(y + radius + padding, image.shape[0])


            # Crop out the rectangular region (bounding box around the circle)
            cropped_gray = gray[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            cropped_gray = gray[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Threshold to binary image
            _, binary_image = cv2.threshold(cropped_gray, 150, 255, cv2.THRESH_BINARY)

            # Morphological closing (fills small holes and connects nearby white regions)
            kernel = np.ones((3, 3), np.uint8)
            closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

            # Count black pixels
            black_pixel_count = np.count_nonzero(closing == 0)
            # mean_intensity = cv2.mean(thresh)[0]

            # Mark buubble as filled if it goes over threshold
            if black_pixel_count > 350:
                filled[i] = {
                    "filled": True,
                    "circle": c,
                    "crossed" : False,
                    "key" : self.answer_key[i]
                }

        return filled


    def combineContours(self, circles, contours):
        final_contours = np.int32(np.around(contours))
        final_circles = [pt for pt in circles[0, :]]
        if len(final_contours) > 0:
            filtered = []
            for c in np.concatenate((final_circles, final_contours)):
                x, _, _ = c
                if x <= 450:
                    filtered.append(c)
            return filtered


    def filterContours(self, cnts):
        filtered_contours = []
        for c in cnts:
            area = cv2.contourArea(c)
            length = cv2.arcLength(c, closed=True)
            if 500 < area and length < 500: 
                (x, y), radius = cv2.minEnclosingCircle(c)
                filtered_contours.append(np.array([x, y, radius]))
        return filtered_contours


    def natural_key(self, filename):
        # Extract numeric value from the filename (e.g., '10.png' -> 10)
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'([0-9]+)', filename)]

    def getUniqueCircles(self, all_circles):
        unique_circles = []
        distance = 0
        for c in all_circles:
            x1, y1, r1 = c
            replace_index = -1
            for j, (x2, y2, r2) in enumerate(unique_circles):
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance <= 25:
                    # Keep the bubble with larger radius
                    if r1 > r2:
                        replace_index = j  # Replace existing with this one
                    break  # Found a match, don't check others

            if replace_index >= 0:
                unique_circles[replace_index] = c
            elif distance > 20 or not unique_circles:  # No match found
                unique_circles.append(c)

        # A bit more filtering to remove any closely located bubbles, threshold set to 80 pixels
        for i in range(len(unique_circles)):
            if(i < (len(unique_circles) - 1)):
                if abs(unique_circles[i][0] - unique_circles[i + 1][0]) < 80:
                    del unique_circles[i]
        return unique_circles
    

    # Helper for extracting page number from a directory
    def extract_page_number(self, folder_name):
        match = re.search(r'\d+', folder_name)
        return int(match.group()) if match else float('inf')

    # Main function for predicting
    def predict(self):
        filenames = [f for f in os.listdir(self.filepath) if f.endswith('.png')]
        filenames = sorted(filenames, key=self.natural_key)
        answers = []
        counter = 0
        for filename in filenames:
            image_path = os.path.join(self.filepath, filename)
            image = cv2.imread(image_path)

            # Resize image to uniform size of 580 by 84 pixels
            image = cv2.resize(image, (580, 84)) 

            # Crop out top and bottom by 10 pixels to remove noise
            cropped = image[10:-10, :]

            # Get the left part (LEtter region)
            left_part = image[:, 12:64]

            # Get the right part which is the bubbles section
            right_part = cropped[:, 65:] 


            gray = cv2.cvtColor(right_part, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            

            # Detect contours and Hough circles
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            detected_circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=15,
                param1=40,
                param2=25,
                minRadius=10,
                maxRadius=40
            )

            
            if detected_circles is not None:
                detected_circles = np.int32(np.around(detected_circles))
            filtered_contours = self.filterContours(cnts)
            all_circles = self.combineContours(detected_circles, filtered_contours)
            unique_circles = self.getUniqueCircles(all_circles)
            unique_circles = sorted(unique_circles, key=lambda x: x[0])
            filled = self.detectFilledInbubbles(unique_circles, gray, image)
            answer = self.predictAnswer(filled, left_part)
            answers.append(answer)

            counter+=1         
            if counter >= self.numQs:
                break

        return answers
