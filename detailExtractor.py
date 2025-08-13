import cv2
import numpy as np
import os
# Class encapsulating logic for extracting questions from the images PNG format
class DetailExtractor:
    def __init__(self, filepath):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = filepath
        # self.filepath = os.path.join(base_dir, "examdetailpages", "page10.png")


    # Extract rectangular regions region of interest (ROI) for each question by cropping out the question from the image
    def crop_question_regions(self, image, column_groups):
        cropped_images = []
    
        horizontal_padding_left = 30
        horizontal_padding_right = 30
        vertical_padding_top = 30
        vertical_padding_bottom = 30
        
        for column in column_groups:
            min_x = min([circle[0] for circle in column])
            max_x = max([circle[0] for circle in column])
            min_y = min([circle[1] for circle in column])
            max_y = max([circle[1] for circle in column])
            
            # Add padding and clamp to image boundaries
            min_x = max(0, min_x - horizontal_padding_left)
            max_x = min(image.shape[1], max_x + horizontal_padding_right)
            min_y = max(0, min_y - vertical_padding_top)
            max_y = min(image.shape[0], max_y + vertical_padding_bottom)
            
            cropped_region = image[min_y:max_y, min_x:max_x]
            cropped_images.append(cropped_region)
        
        return cropped_images



    # Filter detected contours by area
    def filter_contours(self, contours):
        final_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 2000:

                # Get the closing circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                final_contours.append(np.array([x, y, radius]))
        final_contours = np.int32(np.around(final_contours))
        return final_contours
    

    # Get unique circles and remove duplicates
    def get_unique_circles(self, all_circles):
        unique_circles = []
        for c in all_circles:
            x1, y1, _ = c
            too_close = False
            for j in range(len(unique_circles)):
                x2, y2, _ = unique_circles[j]

                # Calculate Euclidean distance between two points
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance <= 15:  # if centers are within 15 pixels remove
                    too_close = True
                    break

            if not too_close:
                unique_circles.append(c)
        return unique_circles
    

    def sort_by_columns(self, data, threshold = 15):
        data = sorted(data, key=lambda x: x[0])

        column_groups = []
        current_column = []

        for c in data:
            if len(current_column) == 0:
                current_column.append(c)
            elif abs(c[0] - current_column[-1][0]) <= threshold:
                current_column.append(c)
            else:
                column_groups.append(current_column)
                current_column = [c]
                    # Add the last column if it exists
        if current_column:
            column_groups.append(current_column)
        
        return column_groups
  
    # Main function extracting the questions from the given image path
    def extract(self):
        # Set threshold for min max y value
        vertical_threshold_min = 300 # Y axis threshold with 10% margin
        vertical_threshold_max = 2300
        horizontal_threshold_min = 640
        horizontal_threshold_max = 2400
        # Load image
        img = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # Apply blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]


        # Filter contours
        final_contours = self.filter_contours(cnts)

        # Filter contours based on Y thresholds
        filtered_contours = [pt for pt in final_contours if (vertical_threshold_min < pt[1] < vertical_threshold_max) and horizontal_threshold_max > pt[0] > horizontal_threshold_min]

        # Find hough circles
        detected_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,       
            param1=40,       
            param2=28,        
            minRadius=15,
            maxRadius=40
        )

        if detected_circles is not None:

            detected_circles = np.int32(np.around(detected_circles))

            # Filter circles based on radius size and y thresholds
            filtered_circles = [pt for pt in detected_circles[0, :] if 17 <= pt[2] <= 35 and (vertical_threshold_min < pt[1] < vertical_threshold_max) and horizontal_threshold_max > pt[0] > horizontal_threshold_min]
            all_circles = []

            # If there is any contours in the pdf combine with the Hough Circles
            if len(filtered_contours) > 0:
                all_circles = np.concatenate((filtered_circles, filtered_contours))
            else:
                all_circles = filtered_circles

            
            all_circles = sorted(all_circles, key=lambda x: x[1])
            # Remove duplicate circle/contour by checking the centres of each circle
            unique_circles = self.get_unique_circles(all_circles)

            # SOrt by groups
            # Student Number
            student_number  = []
            surname = []
            initials = []
            final = []
            for c in unique_circles:
                if c[1] > 400:
                    final.append(c)

            for c in final:
                if 680 < c[0] < 1250:
                    student_number.append(c)
                if 1260 < c[0] < 2100:
                    surname.append(c)
                if 2100 < c[0] < 2350:
                    initials.append(c)
            

            filtered_student_number = []
            for c in student_number:
                if c[1] < 1100:
                    filtered_student_number.append(c)

            # output = img.copy()
            # print(len(initials))
            # for c in initials:
            #     x, y, r = c
            #     cv2.circle(output, (x, y), r, (0, 255, 0), 2)   # green circle
            #     cv2.circle(output, (x, y), 2, (0, 0, 255), 3)    # red center dot

            # display_scale = 0.2
            # display_width = int(img.shape[1] * display_scale)
            # display_height = int(img.shape[0] * display_scale)
            # display_dim = (display_width, display_height)

            # # Resize the output image (with drawings) for display
            # display_img = cv2.resize(output, display_dim, interpolation=cv2.INTER_AREA)

            # cv2.imshow("Detected Circles (Scaled Display)", display_img)  # show the drawn circles image
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            student_number_groups = self.sort_by_columns(filtered_student_number)
            surname_groups = self.sort_by_columns(surname)
            intial_groups = self.sort_by_columns(initials)



            base_dir = os.getcwd()
            os.makedirs(os.path.join(base_dir, "student_number"), exist_ok=True)
            cropped_images = self.crop_question_regions(img, student_number_groups)
            for i, cropped_img in enumerate(cropped_images):
                filename = os.path.join("student_number", f"{i+1}.png")
                # filename = os.path.join("page2", f"{i+1}.png")
                cv2.imwrite(filename, cropped_img)  
            
            base_dir = os.getcwd()
            os.makedirs(os.path.join(base_dir, "surname"), exist_ok=True)
            cropped_images = self.crop_question_regions(img, surname_groups)
            for i, cropped_img in enumerate(cropped_images):
                filename = os.path.join("surname", f"{i+1}.png")
                # filename = os.path.join("page2", f"{i+1}.png")
                cv2.imwrite(filename, cropped_img)  

            base_dir = os.getcwd()
            os.makedirs(os.path.join(base_dir, "initials"), exist_ok=True)
            cropped_images = self.crop_question_regions(img, intial_groups)
            for i, cropped_img in enumerate(cropped_images):
                filename = os.path.join("initials", f"{i+1}.png")
                # filename = os.path.join("page2", f"{i+1}.png")
                cv2.imwrite(filename, cropped_img)  


