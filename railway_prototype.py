import pandas as pd
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
import os
import random

# --- Import libraries for custom dot code functionality ---
from PIL import Image, ImageDraw # Pillow for image creation
import cv2                    # OpenCV for camera access and image processing
import numpy as np            # For numerical operations

# Define the file path for the CSV data
CSV_FILE = r'C:\Users\91887\Desktop\railway\fitting_railway.csv' 
# The directory for our custom codes will be different to avoid confusion
CUSTOM_CODE_DIR = 'custom_codes' 

# --- Configuration for our Custom Dot Code ---
GRID_SIZE = 10       # A 10x10 grid offers 100 positions
NUM_DOTS = 5         # We will use exactly 5 dots for each code
DOT_SIZE = 25        # The size of each cell in pixels
MARGIN = 25          # Margin around the grid
IMAGE_SIZE = GRID_SIZE * DOT_SIZE + 2 * MARGIN # Final image dimensions

class RailwayFittingsPrototype:
    """
    A CLI-based prototype to manage railway fittings data with a
    custom-generated dot code pattern for identification.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        
        if not os.path.exists(CUSTOM_CODE_DIR):
            os.makedirs(CUSTOM_CODE_DIR)
            print(f"Created directory: '{CUSTOM_CODE_DIR}'")

        try:
            self.df = pd.read_csv(self.filepath, sep='\t')
            print(f"Successfully loaded data from '{self.filepath}'.")
            date_columns = ['Manufacturing_Date', 'Last_Inspection_Date', 'Warranty_End_Date']
            for col in [c for c in date_columns if c in self.df.columns]:
                 self.df[col] = pd.to_datetime(self.df[col], format='%d-%m-%Y', errors='coerce')
            
            # Add new columns for our custom dot pattern system if they don't exist
            if 'Custom_Code_File' not in self.df.columns:
                self.df['Custom_Code_File'] = pd.NA
            if 'Dot_Pattern' not in self.df.columns:
                self.df['Dot_Pattern'] = pd.NA


        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            self.df = None

    def _generate_custom_code_image(self, dot_positions, filepath):
        """
        Generates a custom dot code image from a list of dot positions.
        """
        # 1. Create the image
        img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=255) # 'L' for grayscale, 255 for white
        draw = ImageDraw.Draw(img)
        
        # 2. Draw a black border for easier detection
        draw.rectangle([MARGIN - 5, MARGIN - 5, IMAGE_SIZE - MARGIN + 5, IMAGE_SIZE - MARGIN + 5], outline=0, width=5)

        # 3. Draw dots at the specified positions
        for pos in dot_positions:
            row = pos // GRID_SIZE
            col = pos % GRID_SIZE
            
            x0 = MARGIN + col * DOT_SIZE
            y0 = MARGIN + row * DOT_SIZE
            x1 = x0 + DOT_SIZE
            y1 = y0 + DOT_SIZE

            padding = DOT_SIZE * 0.15 
            draw.ellipse([x0 + padding, y0 + padding, x1 - padding, y1 - padding], fill=0)

        img.save(filepath)

    def _scan_custom_code_with_camera(self):
        """
        Opens the webcam to find a dot pattern, read its positions, and search the database.
        """
        print("\nAttempting to open webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Webcam opened. Point a custom code at the camera.")
        print("Press 'q' in the camera window to quit.")
        
        while True:
            success, frame = cap.read()
            if not success: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            found_code = False
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

                if len(approx) == 4 and cv2.contourArea(cnt) > 1000:
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

                    # --- Decoding Logic ---
                    rect = np.zeros((4, 2), dtype="float32")
                    s = approx.sum(axis=2)
                    rect[0] = approx[np.argmin(s)]; rect[2] = approx[np.argmax(s)]
                    diff = np.diff(approx, axis=2)
                    rect[1] = approx[np.argmin(diff)]; rect[3] = approx[np.argmax(diff)]
                    
                    dst = np.array([[0, 0], [IMAGE_SIZE-1, 0], [IMAGE_SIZE-1, IMAGE_SIZE-1], [0, IMAGE_SIZE-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(gray, M, (IMAGE_SIZE, IMAGE_SIZE))

                    # 1. Sample the grid to find dot positions
                    dot_positions = []
                    for i in range(GRID_SIZE * GRID_SIZE):
                        row = i // GRID_SIZE
                        col = i % GRID_SIZE
                        x = MARGIN + int((col + 0.5) * DOT_SIZE)
                        y = MARGIN + int((row + 0.5) * DOT_SIZE)
                        if warped[y, x] < 128:
                            dot_positions.append(i)
                    
                    # 2. If we found the correct number of dots, search for that pattern
                    if len(dot_positions) == NUM_DOTS:
                        # Convert the pattern to a string format for searching
                        pattern_str = ",".join(map(str, sorted(dot_positions)))
                        print(f"\n--- Custom Code Detected! ---\nDetected Pattern: {pattern_str}")
                        self._search_by_dot_pattern(pattern_str)
                        found_code = True
                        break
            
            if found_code:
                break

            cv2.putText(frame, "Scanning... Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Custom Code Scanner", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Scanning cancelled by user.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")
    
    def save_data(self):
        if self.df is not None:
            try:
                df_to_save = self.df.copy()
                date_columns = ['Manufacturing_Date', 'Last_Inspection_Date', 'Warranty_End_Date']
                for col in [c for c in date_columns if c in df_to_save.columns]:
                    df_to_save[col] = pd.to_datetime(df_to_save[col]).dt.strftime('%d-%m-%Y')
                df_to_save.to_csv(self.filepath, index=False, sep='\t')
                print(f"Data successfully saved to '{self.filepath}'.")
            except Exception as e:
                print(f"An error occurred while saving the file: {e}")

    def _display_results(self, results_df, search_term):
        if results_df.empty:
            print(f"\nNo records found for: {search_term}")
        else:
            print(f"\n--- Records Found for: {search_term} ---")
            for index, row in results_df.iterrows():
                print(f"UID: {row['UID']}")
                print(f"  Vendor: {row['Vendor_Name']}")
                print(f"  Batch Code: {row['Batch_Code']}")
                print(f"  Status: {row['Status']}")
                print(f"  Code File: {row.get('Custom_Code_File', 'N/A')}")
                print("-" * 20)

    def _search_by_batch_code(self, batch_code):
        self.df['Batch_Code'] = self.df['Batch_Code'].astype(str)
        results = self.df[self.df['Batch_Code'].str.lower() == batch_code.lower()]
        self._display_results(results, batch_code)
    
    def _search_by_dot_pattern(self, pattern):
        """Searches the DataFrame for a matching dot pattern string."""
        if 'Dot_Pattern' not in self.df.columns:
            print("Error: 'Dot_Pattern' column not found.")
            return
        results = self.df[self.df['Dot_Pattern'] == pattern]
        self._display_results(results, pattern)

    def search_menu(self):
        if self.df is None: return
        while True:
            print("\n--- Search for Records ---")
            print("1. Search Manually (by Lot ID / Batch_Code)")
            print("2. Scan Custom Code via Camera")
            print("3. Back to Main Menu")
            choice = input("Enter your choice (1-3): ").strip()

            if choice == '1':
                batch_code = input("Enter the Batch_Code to search for: ").strip()
                if batch_code: self._search_by_batch_code(batch_code)
            elif choice == '2':
                self._scan_custom_code_with_camera()
            elif choice == '3': break
            else: print("Invalid choice.")

    def add_new_lot(self):
        if self.df is None: return
        print("\n--- Add a New Lot Record ---")
        batch_code = input("Enter Batch Code (e.g., B123): ").strip()
        if not batch_code:
            print("Batch Code cannot be empty."); return
        
        self.df['Batch_Code'] = self.df['Batch_Code'].astype(str)
        if not self.df[self.df['Batch_Code'].str.lower() == batch_code.lower()].empty:
            print(f"\nError: Record with Batch_Code '{batch_code}' already exists.")
            return

        print("\n--- Batch Code is unique. Please enter remaining details. ---")
        vendor_name = input("Vendor Name: ")
        part_type = input("Part Type: ")
        status = input("Status (e.g., OK, FLAGGED): ")
        ai_analysis = input("AI Analysis note: ")

        # --- MODIFIED UID GENERATION (still needed internally) ---
        new_uid = None
        self.df['UID'] = self.df['UID'].astype(str)
        while new_uid is None or not self.df[self.df['UID'] == new_uid].empty:
            vendor_initials = vendor_name[:2].upper()
            current_year = datetime.now().strftime('%y')
            type_initial = part_type[0].upper()
            unique_suffix = str(uuid.uuid4().int)[:4]
            new_uid = f"{vendor_initials}{current_year}{type_initial}{batch_code.replace('B', '')}{unique_suffix}"

        # --- NEW: GENERATE A UNIQUE DOT PATTERN ---
        dot_pattern_str = None
        while dot_pattern_str is None or not self.df[self.df['Dot_Pattern'] == dot_pattern_str].empty:
            # Generate N unique random positions
            positions = random.sample(range(GRID_SIZE * GRID_SIZE), NUM_DOTS)
            # Sort them to create a consistent representation
            positions.sort()
            # Create a string version to store and search
            dot_pattern_str = ",".join(map(str, positions))

        print(f"Generated unique dot pattern: {dot_pattern_str}")
        
        # --- Generate the image from the pattern ---
        code_filepath = os.path.join(CUSTOM_CODE_DIR, f"Code_{new_uid}.png")
        try:
            # Pass the list of positions, not the UID
            self._generate_custom_code_image(positions, code_filepath)
            print(f"Successfully generated custom code to: {code_filepath}")
        except Exception as e:
            print(f"Could not generate custom code. Error: {e}")
            code_filepath = pd.NA

        today_date = pd.to_datetime('today')
        new_record = {
            'UID': new_uid, 'Vendor_Name': vendor_name, 'Part_Type': part_type,
            'Batch_Code': batch_code, 'Manufacturing_Date': today_date,
            'Last_Inspection_Date': today_date,
            'Warranty_End_Date': today_date + pd.DateOffset(years=5),
            'Status': status, 'AI_Analysis': ai_analysis, 
            'Custom_Code_File': code_filepath,
            'Dot_Pattern': dot_pattern_str # Store the new pattern
        }
        
        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
        print(f"\nNew record added successfully with UID: {new_uid}")
        self.save_data()

    def display_dashboard(self):
        if self.df is None: return
        print("\n--- Fittings Analysis Dashboard ---")
        status_counts = self.df['Status'].value_counts()
        print("\nSummary by Manufacturing Status:"); print(status_counts.to_string())
        
        vendor_counts = self.df['Vendor_Name'].value_counts()
        print("\nSummary by Vendor:"); print(vendor_counts.to_string())

        total_records = len(self.df)
        code_generated_count = self.df['Custom_Code_File'].notna().sum()
        print("\nCustom Code Generation Status:")
        print(f"  - {code_generated_count} / {total_records} records have a code.")

        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(8, 6))
            status_counts.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
            ax.set_title('Count of Fittings by Manufacturing Status', fontsize=16)
            ax.set_xlabel('Status'); ax.set_ylabel('Number of Fittings')
            ax.tick_params(axis='x', rotation=0)
            for i, count in enumerate(status_counts):
                ax.text(i, count + 0.5, str(count), ha='center', va='bottom')
            plt.tight_layout()
            print("\nDisplaying status chart. Close chart window to continue...")
            plt.show()
        except Exception as e:
            print(f"\nCould not display plot. Error: {e}")

def main_menu(prototype):
    if prototype.df is None:
        print("\nHalting program due to file loading error.")
        return

    while True:
        print("\n========== Railway Fittings Prototype (Custom Code) ==========")
        print("1. Search for Records")
        print("2. Add a New Lot Record")
        print("3. Display Analysis Dashboard")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1': prototype.search_menu()
        elif choice == '2': prototype.add_new_lot()
        elif choice == '3': prototype.display_dashboard()
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    fittings_prototype = RailwayFittingsPrototype(CSV_FILE)
    main_menu(fittings_prototype)

