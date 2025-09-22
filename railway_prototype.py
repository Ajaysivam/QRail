import pandas as pd
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
import os

# --- Import new libraries for QR functionality ---
import qrcode
from PIL import Image
from pyzbar.pyzbar import decode
import cv2 # Import OpenCV for camera access

# Define the file path for the CSV data
# The user uploaded a file with a similar name. Ensure this matches the file in your directory.
# --- PATH UPDATED AS PER YOUR REQUEST ---
# Note: The file at this path MUST be a .csv file for the program to work.
CSV_FILE = r'C:\Users\91887\Desktop\railway\fitting_railway.csv' 
QR_CODE_DIR = 'qr_codes' # Folder to store generated QR codes

class RailwayFittingsPrototype:
    """
    A CLI-based prototype to manage and analyze railway fittings data
    with QR code generation and scanning support.
    """
    def __init__(self, filepath):
        """
        Initializes the prototype by loading data and setting up the environment.
        """
        self.filepath = filepath
        
        if not os.path.exists(QR_CODE_DIR):
            os.makedirs(QR_CODE_DIR)
            print(f"Created directory: '{QR_CODE_DIR}'")

        try:
            # --- FIX APPLIED HERE ---
            # Added sep='\t' to correctly read the tab-separated file.
            self.df = pd.read_csv(self.filepath, sep='\t')
            print(f"Successfully loaded data from '{self.filepath}'.")

            # --- FIX APPLIED HERE ---
            # Define expected date columns
            date_columns = ['Manufacturing_Date', 'Last_Inspection_Date', 'Warranty_End_Date']
            # Loop only through the columns that actually exist in the CSV to avoid KeyError
            for col in [c for c in date_columns if c in self.df.columns]:
                 self.df[col] = pd.to_datetime(self.df[col], format='%d-%m-%Y', errors='coerce')
            
            if 'QR_Code_File' not in self.df.columns:
                self.df['QR_Code_File'] = pd.NA

        except FileNotFoundError:
            print(f"Error: The file '{self.filepath}' was not found.")
            print("Please make sure your CSV file is in the same directory.")
            self.df = None
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            print("Please ensure the file is a valid tab-separated CSV.")
            self.df = None


    def save_data(self):
        """
        Saves the current state of the DataFrame back to the CSV file.
        """
        if self.df is not None:
            try:
                df_to_save = self.df.copy()
                # --- FIX APPLIED HERE ---
                # Define expected date columns
                date_columns = ['Manufacturing_Date', 'Last_Inspection_Date', 'Warranty_End_Date']
                # Loop only through the columns that actually exist to avoid KeyError
                for col in [c for c in date_columns if c in df_to_save.columns]:
                    # Handle potential NaT values before formatting
                    df_to_save[col] = pd.to_datetime(df_to_save[col]).dt.strftime('%d-%m-%Y')
                
                # --- FIX APPLIED HERE ---
                # Added sep='\t' to save the file in the correct tab-separated format.
                df_to_save.to_csv(self.filepath, index=False, sep='\t')
                print(f"Data successfully saved to '{self.filepath}'.")
            except Exception as e:
                print(f"An error occurred while saving the file: {e}")

    def _display_results(self, results_df, search_term):
        """
        A centralized function to format and print search results.
        """
        if results_df.empty:
            print(f"\nNo records found for: {search_term}")
        else:
            print(f"\n--- Records Found for: {search_term} ---")
            for index, row in results_df.iterrows():
                print(f"UID: {row['UID']}")
                print(f"  Vendor: {row['Vendor_Name']}")
                print(f"  Part Type: {row['Part_Type']}")
                print(f"  Batch Code: {row['Batch_Code']}")
                print(f"  Status: {row['Status']}")
                print(f"  QR File: {row.get('QR_Code_File', 'N/A')}")
                print("-" * 20)

    def _search_by_batch_code(self, batch_code):
        # Ensure Batch_Code is treated as string to prevent errors on .str accessor
        self.df['Batch_Code'] = self.df['Batch_Code'].astype(str)
        results = self.df[self.df['Batch_Code'].str.lower() == batch_code.lower()]
        self._display_results(results, batch_code)

    def _search_by_uid(self, uid):
        self.df['UID'] = self.df['UID'].astype(str)
        results = self.df[self.df['UID'].str.lower() == uid.lower()]
        self._display_results(results, uid)

    def _scan_qr_with_camera(self):
        """
        Opens the webcam to scan for a QR code in real-time.
        """
        print("\nAttempting to open webcam...")
        cap = cv2.VideoCapture(0) # 0 is the default camera

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Webcam opened. Point a QR code at the camera.")
        print("Press 'q' in the camera window to quit.")
        
        qr_found = False
        while not qr_found:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            # Find and decode QR codes
            decoded_objects = decode(frame)
            for obj in decoded_objects:
                # Draw a bounding box around the QR code
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    cv2.polylines(frame, [hull], True, (0, 255, 0), 3)
                else:
                    cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 3)

                uid = obj.data.decode('utf-8')
                print(f"\n--- QR Code Detected! ---")
                print(f"Decoded UID: {uid}")
                self._search_by_uid(uid) # Automatically perform the search
                qr_found = True
                break 

            # Display instructions on the video feed
            cv2.putText(frame, "Scanning... Press 'q' to quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the video feed
            cv2.imshow("QR Code Scanner", frame)

            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Scanning cancelled by user.")
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


    def search_menu(self):
        """
        (Feature 1 Refined) Provides a nested menu for searching.
        """
        if self.df is None:
            print("No data loaded. Cannot perform search.")
            return

        while True:
            print("\n--- Search for Records ---")
            print("1. Search Manually (by Lot ID / Batch_Code)")
            print("2. Scan QR Code via Camera")
            print("3. Back to Main Menu")
            choice = input("Enter your choice (1-3): ").strip()

            if choice == '1':
                batch_code = input("Enter the Batch_Code to search for (e.g., B469): ").strip()
                if batch_code:
                    self._search_by_batch_code(batch_code)
            elif choice == '2':
                self._scan_qr_with_camera() # Call the new camera scanning function
            elif choice == '3':
                print("Returning to main menu...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 3.")

    def add_new_lot(self):
        """
        (Feature 2 Refined) Adds a new lot after checking for duplicates.
        """
        if self.df is None:
            print("No data loaded. Cannot add a new lot.")
            return

        print("\n--- Add a New Lot Record ---")
        
        # Step 1: Check for duplicate Batch_Code first
        batch_code = input("Enter Batch Code (e.g., B123): ").strip()
        if not batch_code:
            print("Batch Code cannot be empty. Aborting.")
            return
        
        self.df['Batch_Code'] = self.df['Batch_Code'].astype(str)
        if not self.df[self.df['Batch_Code'].str.lower() == batch_code.lower()].empty:
            print(f"\nError: Record with Batch_Code '{batch_code}' already exists.")
            print("No new entry created.")
            return

        # Step 2: If unique, proceed to get other details
        print("\n--- Batch Code is unique. Please enter remaining details. ---")
        vendor_name = input("Vendor Name: ")
        part_type = input("Part Type (e.g., Elastic Rail Clip): ")
        status = input("Status (e.g., OK, FLAGGED): ")
        ai_analysis = input("AI Analysis note: ")

        # Step 3: Generate a guaranteed unique UID
        new_uid = None
        self.df['UID'] = self.df['UID'].astype(str)
        while new_uid is None or not self.df[self.df['UID'] == new_uid].empty:
            vendor_initials = vendor_name[:2].upper()
            current_year = datetime.now().strftime('%y')
            type_initial = part_type[0].upper()
            unique_suffix = str(uuid.uuid4().int)[:4]
            new_uid = f"{vendor_initials}{current_year}{type_initial}{batch_code.replace('B', '')}{unique_suffix}"
            if not self.df[self.df['UID'] == new_uid].empty:
                print("Generated UID already exists, regenerating...")

        # Step 4: Generate QR Code
        qr_filepath = os.path.join(QR_CODE_DIR, f"QR_{new_uid}.png")
        try:
            qrcode.make(new_uid).save(qr_filepath)
            print(f"Successfully generated and saved QR code to: {qr_filepath}")
        except Exception as e:
            print(f"Could not generate QR code. Error: {e}")
            qr_filepath = pd.NA

        # Step 5: Create and save the new record
        today_date = pd.to_datetime('today')
        new_record = {
            'UID': new_uid, 'Vendor_Name': vendor_name, 'Part_Type': part_type,
            'Batch_Code': batch_code, 'Manufacturing_Date': today_date,
            'Last_Inspection_Date': today_date,
            'Warranty_End_Date': today_date + pd.DateOffset(years=5),
            'Status': status, 'AI_Analysis': ai_analysis, 'QR_Code_File': qr_filepath
        }
        
        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
        print(f"\nNew record added successfully with UID: {new_uid}")
        self.save_data()

    def display_dashboard(self):
        """
        (Feature 3) Displays summary, chart, and QR code statistics.
        """
        if self.df is None: return
        print("\n--- Fittings Analysis Dashboard ---")
        
        status_counts = self.df['Status'].value_counts()
        print("\nSummary by Manufacturing Status:"); print(status_counts.to_string())
        
        vendor_counts = self.df['Vendor_Name'].value_counts()
        print("\nSummary by Vendor:"); print(vendor_counts.to_string())

        total_records = len(self.df)
        qr_generated_count = self.df['QR_Code_File'].notna().sum()
        missing_count = total_records - qr_generated_count
        print("\nQR Code Generation Status:")
        print(f"  Out of {total_records} total records:")
        print(f"    - {qr_generated_count} QR codes have been generated.")
        print(f"    - {missing_count} are missing QR codes.")

        print("\n--- End of Text Summary ---")
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(8, 6))
            status_counts.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
            ax.set_title('Count of Fittings by Manufacturing Status', fontsize=16)
            ax.set_xlabel('Status', fontsize=12); ax.set_ylabel('Number of Fittings', fontsize=12)
            ax.tick_params(axis='x', rotation=0)
            for i, count in enumerate(status_counts):
                ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=11)
            plt.tight_layout()
            print("\nDisplaying status distribution chart. Close the chart window to continue...")
            plt.show()
        except Exception as e:
            print(f"\nCould not display the plot. Error: {e}")

def main_menu(prototype):
    """
    Displays the main CLI menu and handles user input.
    """
    if prototype.df is None:
        print("\nHalting program due to file loading error.")
        return

    while True:
        print("\n========== Railway Fittings Prototype (QR Enabled) ==========")
        print("1. Search for Records")
        print("2. Add a New Lot Record")
        print("3. Display Analysis Dashboard")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1': prototype.search_menu()
        elif choice == '2': prototype.add_new_lot()
        elif choice == '3': prototype.display_dashboard()
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    fittings_prototype = RailwayFittingsPrototype(CSV_FILE)
    main_menu(fittings_prototype)

