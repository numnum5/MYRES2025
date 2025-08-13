import cv2
import numpy as np
import os
# Class encapsulating logic for extracting questions from the images PNG format
class Extractor:
    def __init__(self, filepath, number_of_pages):
        self.filepath = filepath
        self.number_of_pages = number_of_pages

        # Create directory questions in the current directory to store the questions
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.questions_dir = os.path.join(base_dir, "questions")
        os.makedirs(self.questions_dir, exist_ok=True)

    # Group set of bubbles by questions (req: 5 bubbles next to each other with no more than 150 pixel difference between them) 
    def group_by_columns(self,rows, threshold=150):
        all_column_groups = []
        for row in rows:
            row = sorted(row, key=lambda x: x[0])  # sort by X, acdending order
            columns = []
            current_col = []
            for circle in row:
                if not current_col:
                    current_col.append(circle)
                elif abs(int(circle[0]) - int(current_col[-1][0])) <= threshold:
                    current_col.append(circle)
                else:
                    columns.append(current_col)
                    current_col = [circle]
            if current_col:
                columns.append(current_col)
            all_column_groups.append(columns)
        return all_column_groups
    

    # Extract rectangular regions region of interest (ROI) for each question by cropping out the question from the image
    def crop_question_regions(self, image, column_groups):
        cropped_images = []
        for i in range(len(column_groups[0])):
            for row in column_groups:
                question_group = row[i]
                min_x = min([circle[0] for circle in question_group])  
                max_x = max([circle[0] for circle in question_group]) 
                min_y = min([circle[1] for circle in question_group])
                max_y = max([circle[1] for circle in question_group])
                min_x = max(0, min_x - 92)
                max_x = min(image.shape[1], max_x + 102)
                min_y = max(0, min_y - 40)
                max_y = min(image.shape[0], max_y + 40)
                cropped_region = image[min_y:max_y, min_x:max_x]
                cropped_images.append(cropped_region)
        return cropped_images
    
    # Group set of bubbles by row (simlar y coords with margin of 30 pixels default)
    def group_by_rows(self, circles, threshold=30):
        rows = []
        current_row = []
        for circle in circles:
            if not current_row:
                current_row.append(circle)
            elif abs(circle[1] - current_row[-1][1]) <= threshold:
                current_row.append(circle)
            else:
                rows.append(current_row)
                current_row = [circle]
        if current_row:
            rows.append(current_row)    
        return rows


    # Filter detected contours by area
    def filter_contours(self, contours):
        final_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area:

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
    

    def fill_missing_bubbles(self, filtered_da, expected_x_positions=None, tolerance=20):
        """
        Ensures each row has bubbles at the expected X positions by inserting placeholders if missing.

        :param filtered_da: List of lists of bubbles [(x, y, radius), ...] per row
        :param expected_x_positions: List of expected X coordinates for each bubble
        :param tolerance: Pixel tolerance for matching expected X positions
        :return: List of rows with all expected positions filled
        """
        if expected_x_positions is None:
            expected_x_positions = [
                192, 288, 382, 474, 572, 784, 880, 974, 1066, 1162,
                1374, 3214, 1470, 1564, 1658, 1752, 1966, 2060, 2156, 2248
            ]

        final = []
        for row in filtered_da:
            r = sorted(row, key=lambda x: x[0])
            filled_row = []

            if len(r) < len(expected_x_positions):
                for expected_x in expected_x_positions:
                    match = next((bubble for bubble in r if abs(bubble[0] - expected_x) <= tolerance), None)
                    if match is not None:
                        filled_row.append(match)
                    else:
                        avg_y = sum(b[1] for b in r) // len(r)
                        filled_row.append(np.array([expected_x, avg_y, 22]))
            else:
                filled_row = r

            final.append(filled_row)

        return final

    def filter_detected_bubbles(self, filtered_rows):
        """
        Filters out unwanted bubbles from rows based on their X-coordinate positions.

        Rules:
        - Remove any bubbles detected between specific column group ranges:
            (610, 740), (1200, 1340), (1800, 1900)
        - Remove any bubbles with x < 140

        :param filtered_rows: List of lists, where each sublist contains bubble coordinates (x, y, ...)
        :return: A new list with filtered bubbles
        """
        filtered_bubbles = []
        for row in filtered_rows:
            filtered_row = []
            for element in row:
                x = int(element[0])
                if not (1200 < x < 1340) and not (610 < x < 740) and not (1800 < x < 1900):
                    filtered_row.append(element)
            filtered_bubbles.append(filtered_row)

        # Remove any bubbles less than 140 pixels from the left
        filtered_bubbles = [
            [element for element in row if element[0] >= 140]
            for row in filtered_bubbles
        ]

        return filtered_bubbles
    
    def filter_valid_groups(self, final, min_vertical_spacing=86):
        """
        Filters groups so that consecutive accepted groups are at least
        `min_vertical_spacing` pixels apart in the Y-axis.

        :param final: List of groups, where each group is a list of bubbles [(x, y, radius), ...]
        :param min_vertical_spacing: Minimum vertical distance between groups
        :return: List of valid groups
        """
        valid_groups = []
        prev_y = None

        for group in final:
            if not group:  # Skip empty groups
                continue

            y = group[0][1]  # Y-coordinate of the first bubble in the group
            if prev_y is None:
                valid_groups.append(group)
                prev_y = y
            else:
                diff = abs(y - prev_y)
                if diff >= min_vertical_spacing:
                    valid_groups.append(group)
                    prev_y = y

        return valid_groups
    # Main function extracting the questions from the given image path
    def extract(self):
    
        # Iterate over the number of pages
        for page_number in range(0, self.number_of_pages):

            # Set threshold for min max y value
            vertical_threshold_min = 2300 # Y axis threshold with 10% margin
            vertical_threshold_max = 3300 

            # If it's the second page set the y to 200 to start from the top of the page
            if(page_number >= 1):
                vertical_threshold_min = 200


            # Load image
            img_path = os.path.join(self.filepath, f"page{page_number + 1}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
            filtered_contours = [pt for pt in final_contours if (vertical_threshold_min < pt[1] < vertical_threshold_max)]

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
                filtered_circles = [pt for pt in detected_circles[0, :] if 17 <= pt[2] <= 35 and (vertical_threshold_min < pt[1] < vertical_threshold_max)]
                all_circles = []

                # If there is any contours in the pdf combine with the Hough Circles
                if len(filtered_contours) > 0:
                    all_circles = np.concatenate((filtered_circles, filtered_contours))
                else:
                    all_circles = filtered_circles

                
                all_circles = sorted(all_circles, key=lambda x: x[1])

                # Remove duplicate circle/contour by checking the centres of each circle
                unique_circles = self.get_unique_circles(all_circles)
                # Organise the bubbles by rows
                rows = self.group_by_rows(unique_circles)

                # Filter rows by removing duplicates or bubbles that are too close too each other
                filtered_rows = []
                current = []
                for row in rows:
                    current = row
                    if len(row) > 20:
                        current = sorted(row, key=lambda x: x[0])
                        for i in range(len(current)):
                            if(i < (len(current) - 1)):
                                # Removes any duplciate or close bubbles that is less than 80 distance appart
                                if abs(current[i][0] - current[i + 1][0]) < 80:
                                    if 610 < int(current[i][0]) < 740:
                                        del current[i]
                                    elif 610 < int(current[i + 1][0]) < 740:
                                        del current[i + 1]
                                    elif 1200 < int(current[i][0]) < 1340:
                                        del current[i]
                                    elif 1200 < int(current[i + 1][0]) < 1340:
                                        del current[i + 1]
                                    elif 1800 < int(current[i][0]) < 1900:
                                        del current[i]
                                    elif 1800 < int(current[i + 1][0]) < 1900:
                                        del current[i + 1]
                    filtered_rows.append(current)
                
                # after doing this i gotta check if its detecting something on far left
                filtered_data = self.filter_detected_bubbles(filtered_rows)
                final = self.fill_missing_bubbles(filtered_data)
                final.sort(key=lambda g: g[0][1])
                valid_groups = self.filter_valid_groups(final)


                # output = img.copy()
                # for v in valid_groups:
                #     for c in v:
                #         x, y, r = c
                #         cv2.circle(output, (x, y), r, (0, 255, 0), 2)   # green circle
                #         cv2.circle(output, (x, y), 2, (0, 0, 255), 3)    # red center dot

                # display_scale = 0.2
                # display_width = int(img.shape[1] * display_scale)
                # display_height = int(img.shape[0] * display_scale)
                # display_dim = (display_width, display_height)

                # # Resize the output image (with drawings) for display
                # display_img = cv2.resize(output, display_dim, interpolation=cv2.INTER_AREA)

                # cv2.imshow("Detected Circles (Scaled Display)", display_img)  # show the drawn circles image
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()


                bubble_grid = self.group_by_columns(valid_groups)
                
                cropped_images = self.crop_question_regions(img, bubble_grid)

                
                prefix = 1
                if page_number + 1 > 1:
                    prefix = 41
                for page_number, cropped_img in enumerate(cropped_images):
                    # print(page_number + prefix)
                    filename = os.path.join(self.questions_dir, f"{page_number + prefix}.png")
                    cv2.imwrite(filename, cropped_img) 
        return self.questions_dir
