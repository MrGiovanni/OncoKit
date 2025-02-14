"""
FEATURES:
1. Extracts metadata from an Excel file for each BDMAP ID (folder).
2. Processes CT scan data and segmentation masks to create overlay and zoomed-in images.
3. Dynamically generates PDF reports for each folder using a provided template.
4. Removes blank pages from the generated PDFs.
5. Handles multiple folders automatically using command-line arguments.

- Required libraries: SimpleITK, nibabel, numpy, matplotlib, pandas, argparse, reportlab, PyPDF2, openpyxl,tqdm,

- Workflow
Read Input Data: The script reads the metadata and segmentation details from the provided Excel file.
Prepare Tasks: Each folder is treated as an individual task, and tasks are prepared based on the metadata.
Process Folders:
Check inclusion criteria for segmentation masks.
Validate the presence of CT files.
Generate PDF reports with relevant sections.
Skip the Key Images section if no valid masks are found, but still generate the report.
Output: Saves the reports as PDF files in the specified output directory.
Error Handling: Logs errors to error_log.txt without interrupting the processing of other folders, error_log.tex in output_folder.
USAGE:

-Run this script from the command line with the following arguments:

    python <script_name>.py --excel_file <EXCEL_FILE> --base_folder <BASE_FOLDER> --output_dir <OUTPUT_DIR> --template_pdf <TEMPLATE_PDF>
    python medical_report_generation.py --excel_file /mnt/realccvl15/zzhou82/project/OncoKit/utils/data_demo/AbdomenAtlas3.0.csv --base_folder /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlasX/AbdomenAtlasX --output_dir /mnt/T8/error_analysis/PDF_Report --template_pdf /mnt/realccvl15/zzhou82/project/OncoKit/utils/data_demo/PDF_template.pdf


ARGUMENTS:
1. `--excel_file`: Path to the Excel file containing metadata (must include a 'BDMAP ID' column).
2. `--base_folder`: Base directory containing folders for each BDMAP ID. Each folder must include:
   - `ct.nii.gz`: The CT scan file.
   - Segmentation mask files in a `segmentations/` subfolder (`liver_lesion.nii.gz`, `pancreatic_lesion.nii.gz`, and `kidney_lesion.nii.gz`).
3. `--output_dir`: Directory where the generated PDF reports will be saved.
4. `--template_pdf`: Path to a blank PDF template used for creating the reports.

"""

import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter, PageObject
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing 
from functools import partial
from PIL import Image

def validate_image(image_path):
    """
    Check if an image file is valid and can be opened.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if valid, False if corrupt or missing.
    """
    if not os.path.exists(image_path):
        return False  # File is missing

    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify integrity
        return True  # Image is valid
    except Exception:
        return False  # Image is corrupt


# Step 1: Read Excel and Filter Information
def read_excel(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found: {}".format(file_path))

    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path, low_memory=False)  # Suppress DtypeWarning
    elif file_path.endswith((".xlsx", ".xls")):
        data = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Expected .csv, .xlsx, or .xls.")

    if "BDMAP ID" not in data.columns:
        raise ValueError("The file must contain a 'BDMAP ID' column.")

    return data

def reorient_to_ras(nifti_image):
    """
    Reorient a NIfTI image to RAS (Right-Anterior-Superior) orientation.
    This ensures consistent orientation for processing.
    """
    try:
        ras_image = nib.as_closest_canonical(nifti_image)  # Convert to RAS orientation
        return ras_image.get_fdata()  # Return the data array
    except Exception as e:
        raise ValueError(f"Error reorienting to RAS: {e}")

# Helper Function to Process CT and Mask
def get_most_labeled_slice(ct_path, mask_path, output_png, contrast_min=-150, contrast_max=250):
    """
    Load CT and mask, ensure RAS orientation, find the most labeled slice, and generate an overlay image.
    """
    try:
        # Load the CT scan and mask
        ct_scan = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)

        # Reorient to RAS
        ct_scan = sitk.DICOMOrient(ct_scan, 'RAS')
        mask = sitk.DICOMOrient(mask, 'RAS')

        # Convert to numpy arrays
        ct_array = sitk.GetArrayFromImage(ct_scan)
        mask_array = sitk.GetArrayFromImage(mask)

        # Check for shape mismatches
        if ct_array.shape != mask_array.shape:
            raise ValueError(f"Shape mismatch: CT shape {ct_array.shape}, Mask shape {mask_array.shape}")

        # Find the slice with the most labels
        slice_sums = np.sum(mask_array, axis=(1, 2))
        most_labeled_slice_index = np.argmax(slice_sums)

        # Get the CT and mask slices
        ct_slice = ct_array[most_labeled_slice_index]
        mask_slice = mask_array[most_labeled_slice_index]

        # Apply mirroring
        ct_slice = np.fliplr(ct_slice)
        mask_slice = np.fliplr(mask_slice)

        # Apply contrast adjustment
        ct_slice = np.clip(ct_slice, contrast_min, contrast_max)
        ct_slice = (ct_slice - contrast_min) / (contrast_max - contrast_min) * 255
        ct_slice = ct_slice.astype(np.uint8)

        # Overlay mask contours on CT slice
        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap='gray', origin='lower')
        plt.contour(mask_slice, colors='red', linewidths=1)  # Use red contours for the mask
        plt.axis('off')
        plt.savefig(output_png, bbox_inches="tight", pad_inches=0)
        plt.close()
        return True
    except Exception:
        return False

def create_overlay_image(ct_path, mask_path, output_path, color="red"):
    """
    Generate overlay images for most labeled slices using the unified RAS orientation logic.
    """
    try:
        return get_most_labeled_slice(ct_path, mask_path, output_path)
    except Exception as e:
        return False

# Helper Function to Zoom into Labeled Area
def zoom_into_labeled_area(ct_path, mask_path, output_path, color="red"):
    """
    Create a zoomed-in view of the largest labeled area with consistent RAS orientation.
    """
    try:
        # Load the CT scan and mask
        ct_scan = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)

        # Reorient to RAS
        ct_scan = sitk.DICOMOrient(ct_scan, 'RAS')
        mask = sitk.DICOMOrient(mask, 'RAS')

        # Convert to numpy arrays
        ct_array = sitk.GetArrayFromImage(ct_scan)
        mask_array = sitk.GetArrayFromImage(mask)

        # Check for shape mismatches
        if ct_array.shape != mask_array.shape:
            raise ValueError(f"Shape mismatch: CT shape {ct_array.shape}, Mask shape {mask_array.shape}")

        # Find the slice with the most labels
        slice_sums = np.sum(mask_array, axis=(1, 2))
        largest_slice_idx = np.argmax(slice_sums)
        if slice_sums[largest_slice_idx] == 0:
            raise ValueError("No labeled area found in the mask.")

        # Get the mask slice and calculate the bounding box
        mask_slice = mask_array[largest_slice_idx]
        coords = np.array(np.where(mask_slice))
        min_row, max_row = np.min(coords[0]), np.max(coords[0])
        min_col, max_col = np.min(coords[1]), np.max(coords[1])
        padding = 20
        min_row = max(min_row - padding, 0)
        max_row = min(max_row + padding, mask_slice.shape[0])
        min_col = max(min_col - padding, 0)
        max_col = min(max_col + padding, mask_slice.shape[1])

        # Extract the zoomed region
        zoomed_image = ct_array[largest_slice_idx][min_row:max_row, min_col:max_col]
        zoomed_mask = mask_array[largest_slice_idx][min_row:max_row, min_col:max_col]

        # Apply mirroring
        zoomed_image = np.fliplr(zoomed_image)
        zoomed_mask = np.fliplr(zoomed_mask)

        # Apply contrast adjustment to the zoomed CT slice
        zoomed_image = np.clip(zoomed_image, -150, 250)
        zoomed_image = (zoomed_image + 150) / 400 * 255
        zoomed_image = zoomed_image.astype(np.uint8)

        # Save the zoomed-in image with overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(zoomed_image, cmap="gray", origin="lower")
        plt.contour(zoomed_mask, colors=color, linewidths=1)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        return False

# PDF Generation
def draw_table(pdf, table_data, x, y, total_width, row_height, bold=False):
    """
    Draws a table in the PDF with a dynamically calculated width to align with text margins.
    """
    num_columns = len(table_data[0])  # Number of columns in the table
    col_width = total_width / num_columns  # Calculate column width

    for row in table_data:
        x_pos = x
        for col_index, cell in enumerate(row):
            pdf.rect(x_pos, y - row_height, col_width, row_height)
            pdf.setFont("Helvetica-Bold" if bold else "Helvetica", 10)

            # Handle NaN and convert other numeric values to integers where applicable
            if pd.isna(cell):
                cell = "N/A"
            elif isinstance(cell, (float, int)):
                cell = int(cell) if col_index == 2 and cell == cell else cell  # Convert lesion count to integer

            pdf.drawString(x_pos + 5, y - row_height + 5, str(cell))
            x_pos += col_width
        y -= row_height

        # Handle page overflow
        if y < 50:
            pdf.showPage()
            y = letter[1] - 100
    return y

def start_new_page(pdf, height, top_margin=100):
    """
    Start a new page in the PDF and reset the y_position.
    Args:
        pdf: The canvas object for the PDF.
        height: The height of the PDF page.
        top_margin: The top margin to leave on the new page.
    Returns:
        y_position: The reset y_position for the new page.
    """
    pdf.showPage()  # Start a new page
    pdf.setFont("Helvetica", 12)  # Reset font to default
    y_position = height - top_margin  # Reset y_position
    return y_position

def extract_and_format_number(value):
    """
    Extracts the numeric part from a string, rounds it to 1 decimal, and converts to a float.
    Handles content with brackets or extra text (e.g., '34.567 (note)').

    Args:
        value (str): The content to process.
    
    Returns:
        float: Rounded numeric value with one decimal.
        str: 'N/A' if no valid numeric value is found.
    """
    if pd.isna(value):  # Handle NaN values
        return "N/A"
    
    try:
        # Extract the first number from the string (handles brackets and extra text)
        import re
        number_match = re.search(r"[-+]?\d*\.?\d+", str(value))
        if number_match:
            number = float(number_match.group())  # Extract numeric part
            return round(number, 1)  # Round to one decimal
        else:
            return "N/A"  # No numeric part found
    except Exception:
        return "N/A"  # Handle unexpected errors

def generate_pdf_with_template(
    output_pdf, folder_name, extracted_data, column_headers, ct_path, masks,
    template_pdf, temp_pdf_path
):
    """
    Generate a PDF report using a blank PDF template for each page.

    Args:
        output_pdf (str): Path to save the final output PDF.
        folder_name (str): Folder name being processed.
        extracted_data (pd.Series): Data extracted for this folder.
        column_headers (list): Headers from the input data file.
        ct_path (str): Path to the CT file.
        masks (dict): Paths to valid segmentation masks.
        template_pdf (str): Path to the blank template PDF.
        temp_pdf_path (str): Path to save the temporary content PDF (unique per process).
    """
    try:
        temp_pdf = canvas.Canvas(temp_pdf_path, pagesize=letter)
        width, height = letter
        left_margin, top_margin = 50, 100
        line_height, section_spacing = 12, 30
        total_table_width = width - 2 * left_margin
        y_position = height - top_margin

        def reset_page():
            nonlocal y_position
            temp_pdf.showPage()
            y_position = height - 120
            temp_pdf.setFont("Helvetica", 10)

        def check_and_reset_page(space_needed):
            nonlocal y_position
            if y_position - space_needed < 50:
                reset_page()

        def write_wrapped_text(x, y, content, bold=False, font_size=10, max_width=None):
            temp_pdf.setFont("Helvetica-Bold" if bold else "Helvetica", font_size)
            words = content.split()
            current_line = ""
            max_width = max_width or width - left_margin * 2
            for word in words:
                if temp_pdf.stringWidth(current_line + word + " ", "Helvetica", font_size) > max_width:
                    temp_pdf.drawString(x, y, current_line.strip())
                    y -= line_height
                    current_line = f"{word} "
                    if y < 50:
                        reset_page()
                        y = height - top_margin
                else:
                    current_line += f"{word} "
            if current_line:
                temp_pdf.drawString(x, y, current_line.strip())
                y -= line_height
            return y

     

        temp_pdf.setFont("Helvetica-Bold", 26)  # Set font to bold and large

    # Calculate the width of the text to center it
        text_width = temp_pdf.stringWidth("MEDICAL REPORT", "Helvetica-Bold", 26)
        center_x = (width - text_width) / 2  # Calculate x-coordinate to center the text

    # Adjust y_position for custom placement of the title
        y_position = height - 130  # Adjust 100 to control the distance from the top

    # Draw the centered title
        temp_pdf.drawString(center_x, y_position, "MEDICAL REPORT")
        y_position -= 30  # Adjust spacing below the title 

    # Replace all instances of data with table_data in the context of draw_table.
        # Section 1a: Patient Information
        temp_pdf.setFont("Helvetica-Bold", 12)
        y_position -= 0
        temp_pdf.drawString(left_margin, y_position, "PATIENT INFORMATION")
        y_position -= line_height

        # Replace NaN values with "N/A" in patient information
        left_content = [
            ["BDMAP ID", folder_name],
            ["Age", int(extracted_data["age"]) if not pd.isna(extracted_data["age"]) else "N/A"]
    ]
        right_content = [
            ["Sex", extracted_data["sex"] if not pd.isna(extracted_data["sex"]) else "N/A"]
        ]

        left_y, right_y = y_position, y_position
        for item in left_content:
            left_y = write_wrapped_text(left_margin, left_y, f"{item[0]}: {item[1]}")
        for item in right_content:
            right_y = write_wrapped_text(width / 2, right_y, f"{item[0]}: {item[1]}")

        y_position = min(left_y, right_y) - section_spacing

        # Section 1b: Imaging Details
        temp_pdf.setFont("Helvetica-Bold", 12)
        temp_pdf.drawString(left_margin, y_position, "IMAGING DETAIL")
        y_position -= line_height

        # Define column indices for imaging details
        imaging_left = [1, 2]  # Example: Columns B and C for the left side
        imaging_right = [5, 6]  # Example: Other imaging details for the right side

        # Process imaging details for columns B and C (left side)
        left_y = y_position
        right_y = y_position

        for col in imaging_left:
            header, value = column_headers[col], extracted_data.iloc[col]
            if col == 1:  # Special handling for column index B
                if pd.notna(value):  # Check if the value is not NaN
                    try:
                        # Process column B: Extract numbers, round each to 1 decimal, and format as a list
                        value = str(value)  # Ensure value is string
                        numbers = [float(num) for num in value.replace('[', '').replace(']', '').split() if num.replace('.', '', 1).isdigit()]
                        if numbers:
                            # Round each number to 1 decimal place
                            processed_numbers = [round(num, 1) for num in numbers]
                            value = f"[{' '.join(map(str, processed_numbers))}]"
                        else:
                            value = "N/A"  # Default if no valid number is found
                    except ValueError:
                        value = "N/A"  # Handle invalid numeric conversion
                else:
                    value = "N/A"  # Default if NaN
            else:  # For column C or any other column
                value = value if pd.notna(value) else "N/A"  # Default to "N/A" if value is NaN

            # Format and display the content for the left side
            content = f"{header}: {value}"
            left_y = write_wrapped_text(left_margin, left_y, content)

        # Process imaging details for the right side (e.g., columns E and F)
        for col in imaging_right:
            header, value = column_headers[col], extracted_data.iloc[col]
            content = f"{header}: {value if pd.notna(value) else 'N/A'}"
            right_y = write_wrapped_text(width / 2, right_y, content)

        # Update the y_position after processing imaging details
        y_position = min(left_y, right_y) - section_spacing

        # Section 2: AI Measurements
        temp_pdf.setFont("Helvetica-Bold", 12)
        temp_pdf.drawString(left_margin, y_position, "AI MEASUREMENTS")
        y_position -= line_height
        table_data = [
            ["", "organ volume (cc)", "total lesion #", "total lesion volume (cc)"], 
            [
                "liver", 
                extracted_data.iloc[7], 
                "N/A" if pd.isna(extracted_data.iloc[11]) else int(extracted_data.iloc[11]), 
                extracted_data.iloc[8]
            ],
            [
                "pancreas", 
                extracted_data.iloc[23], 
                "N/A" if pd.isna(extracted_data.iloc[27]) else int(extracted_data.iloc[27]), 
                extracted_data.iloc[24]
            ],
            [
                "kidney", 
                extracted_data.iloc[40], 
                "N/A" if pd.isna(extracted_data.iloc[46]) else int(extracted_data.iloc[46]), 
                extracted_data.iloc[43]
            ],
        ]
        row_height = 30
        y_position = draw_table(temp_pdf, table_data, left_margin, y_position, total_table_width, row_height, bold=False)
        y_position -= section_spacing

        # Section 3: Narrative Report
        temp_pdf.setFont("Helvetica-Bold", 12)
        temp_pdf.drawString(left_margin, y_position, "NARRATIVE REPORT")
        y_position -= line_height
        narrative_report = str(extracted_data.iloc[71])
        for line in narrative_report.split("\n"):
            y_position = write_wrapped_text(left_margin, y_position, line.strip())
            if y_position < 50:
                temp_pdf.showPage()
                y_position = height - top_margin
        y_position -= section_spacing

        # Section 4: Structured Report
        temp_pdf.setFont("Helvetica-Bold", 12)
        temp_pdf.drawString(left_margin, y_position, "STRUCTURED REPORT")
        y_position -= line_height
        structured_report = str(extracted_data.iloc[70])
        for line in structured_report.split("\n"):
            y_position = write_wrapped_text(left_margin, y_position, line.strip())
            if y_position < 50:
                reset_page()
        y_position -= section_spacing

        # Section 5: Key Images
        include_liver = not (pd.isna(extracted_data.iloc[11]) or extracted_data.iloc[11] == 0)
        include_pancreas = not (pd.isna(extracted_data.iloc[27]) or extracted_data.iloc[27] == 0)
        include_kidney = not (pd.isna(extracted_data.iloc[46]) or extracted_data.iloc[46] == 0)

        if include_liver or include_pancreas or include_kidney:
            y_position = start_new_page(temp_pdf, height)
            temp_pdf.setFont("Helvetica-Bold", 14)
            temp_pdf.drawString(left_margin, y_position, "KEY IMAGES")
            y_position -= section_spacing

            for organ, mask_path in masks.items():
                if (organ == "liver" and not include_liver) or \
                (organ == "pancreas" and not include_pancreas) or \
                (organ == "kidney" and not include_kidney):
                    continue

                header = f"{organ.upper()} TUMORS"
                check_and_reset_page(space_needed=line_height)

                temp_pdf.setFont("Helvetica", 12)
                temp_pdf.drawString(left_margin, y_position, header)
                y_position -= line_height

                # Check space for the first image (overlay image)
                check_and_reset_page(space_needed=220)

                # Add overlay image
                if create_overlay_image(ct_path, mask_path, f"/tmp/{organ}_overlay.png", color="red"):
                    temp_pdf.drawImage(f"/tmp/{organ}_overlay.png", left_margin, y_position - 200, width=200, height=200)

                    # Check space for the second image (zoomed image)
                    check_and_reset_page(space_needed=220)

                    # Add zoomed image
                    zoom_success = zoom_into_labeled_area(ct_path, mask_path, f"/tmp/{organ}_zoomed.png", color="red")
                    if zoom_success:
                        temp_pdf.drawImage(f"/tmp/{organ}_zoomed.png", left_margin + 250, y_position - 205, width=210, height=210)

                    # Update y_position after the images
                    y_position -= 220

        temp_pdf.save() 

        template_reader = PdfReader(template_pdf)
        content_reader = PdfReader(temp_pdf_path)
        writer = PdfWriter()

        for page in content_reader.pages:
            template_page = template_reader.pages[0]  # Use the first page of the template
            merged_page = PageObject.create_blank_page(width=template_page.mediabox.width, height=template_page.mediabox.height)
            merged_page.merge_page(template_page)
            merged_page.merge_page(page)
            writer.add_page(merged_page)

        with open(output_pdf, "wb") as final_pdf:
           writer.write(final_pdf)

    except Exception as e:
        raise RuntimeError(f"Error generating PDF for {folder_name}: {e}")

    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def remove_blank_pages(pdf_path):
    """
    Removes blank pages from a PDF and overwrites the original file.

    Args:
        pdf_path (str): Path to the PDF file to clean.
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for page in reader.pages:
        # Check if the page has content (non-blank)
        if page.extract_text().strip():  # Non-empty text indicates a non-blank page
            writer.add_page(page)

    # Overwrite the original PDF with the cleaned version
    with open(pdf_path, "wb") as output_file:
        writer.write(output_file)


def prepare_task(row, base_folder, column_headers, args):
    """
    Prepares a single task for a folder.

    Args:
        row (pd.Series): The row of data corresponding to the folder.
        base_folder (str): The base folder containing the data.
        column_headers (list): Column headers of the Excel file.
        args (Namespace): The command-line arguments.

    Returns:
        tuple: The task arguments for `process_folder`, or None if the folder does not exist.
    """
    folder_name = row["BDMAP ID"]
    folder_path = os.path.join(base_folder, folder_name)

    if os.path.exists(folder_path):
        return folder_name, folder_path, row, column_headers, args
    else:
        return None

def process_folder(task):
    """
    Process a single folder to generate the medical report PDF.

    Args:
        task (tuple): Contains folder_name, folder_path, row, column_headers, args.

    Returns:
        str: Status message for the processed folder.
    """
    folder_name, folder_path, row, column_headers, args = task

    # Define error log path in the output directory
    error_log_path = os.path.join(args.output_dir, "error_log.txt")

    try:
        # Step 1: Check mask inclusion based on BDMAP data
        include_liver = not (pd.isna(row.iloc[11]) or row.iloc[11] == 0)
        include_pancreas = not (pd.isna(row.iloc[27]) or row.iloc[27] == 0)
        include_kidney = not (pd.isna(row.iloc[46]) or row.iloc[46] == 0)

        # Step 2: Define paths for the CT file and masks
        ct_path = os.path.join(folder_path, "ct.nii.gz")
        masks = {
            "liver": os.path.join(folder_path, "segmentations", "liver_lesion.nii.gz") if include_liver else None,
            "pancreas": os.path.join(folder_path, "segmentations", "pancreatic_lesion.nii.gz") if include_pancreas else None,
            "kidney": os.path.join(folder_path, "segmentations", "kidney_lesion.nii.gz") if include_kidney else None,
        }

        # Step 3: Remove missing or invalid masks
        masks = {organ: path for organ, path in masks.items() if path and os.path.exists(path)}

        # Step 4: Generate overlay and zoomed-in images
        image_paths = {}
        corrupt_images = []

        for organ, mask_path in masks.items():
            overlay_path = f"/tmp/{folder_name}_{organ}_overlay.png"
            zoomed_path = f"/tmp/{folder_name}_{organ}_zoomed.png"

            overlay_success = create_overlay_image(ct_path, mask_path, overlay_path)
            zoom_success = zoom_into_labeled_area(ct_path, mask_path, zoomed_path)

            # Validate generated images
            if overlay_success and validate_image(overlay_path):
                image_paths[f"{organ}_overlay"] = overlay_path
            else:
                corrupt_images.append(f"{organ} overlay image is missing or corrupt")

            if zoom_success and validate_image(zoomed_path):
                image_paths[f"{organ}_zoomed"] = zoomed_path
            else:
                corrupt_images.append(f"{organ} zoomed-in image is missing or corrupt")

        # Step 5: Log issues if images are missing or corrupt
        if corrupt_images:
            with open(error_log_path, "a") as log_file:
                log_file.write(f"Skipping Key Images section for {folder_name} due to errors:\n")
                for error_msg in corrupt_images:
                    log_file.write(f"  - {error_msg}\n")
            masks = None  # Skip Key Images section

        # Step 6: Define output PDF paths
        output_pdf = os.path.join(args.output_dir, f"{folder_name}.pdf")
        temp_pdf_path = os.path.join(args.output_dir, f"temp_{folder_name}.pdf")

        # Step 7: Generate the PDF (excluding Key Images if masks are None)
        generate_pdf_with_template(
            output_pdf=output_pdf,
            folder_name=folder_name,
            extracted_data=row,
            column_headers=column_headers,
            ct_path=ct_path,
            masks=masks if not corrupt_images else None,  # Skip Key Images if errors
            template_pdf=args.template_pdf,
            temp_pdf_path=temp_pdf_path,
        )

        return f"Successfully processed folder: {folder_name}"

    except Exception as e:
        # Log errors in the output directory
        with open(error_log_path, "a") as log_file:
            log_file.write(f"Error processing folder {folder_name}: {str(e)}\n")
        return f"Error processing folder {folder_name}: {e}"

    finally:
        # Step 8: Cleanup - remove temp PDF and temporary images
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        
        for image_path in image_paths.values():
            if os.path.exists(image_path):
                os.remove(image_path)

def main(args):
    """
    Main function to process multiple folders using multiprocessing.
    """
    os.makedirs(args.output_dir, exist_ok=True)  # Ensure output directory exists
    error_log_path = os.path.join(args.output_dir, "error_log.txt")  # Log file in output directory

    # Clear previous error log
    open(error_log_path, "w").close()  # Overwrite with an empty file

    try:
        data = read_excel(args.excel_file)
        num_cores = args.num_core if args.num_core > 0 else min(multiprocessing.cpu_count(), 16)

        # 🔥 STEP 1: PARALLEL TASK PREPARATION WITH PROGRESS BAR
        tasks = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {
                executor.submit(prepare_task, row, args.base_folder, data.columns, args): row["BDMAP ID"]
                for _, row in data.iterrows()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing Tasks", ncols=80):
                task = future.result()
                if task:
                    tasks.append(task)  # Store valid tasks

        if not tasks:
            return  # No valid folders found, exit quietly

        # 🔥 STEP 2: PARALLEL PROCESSING OF FOLDERS WITH PROGRESS BAR
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(process_folder, task): task[0] for task in tasks}

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Folders", ncols=80):
                pass  # No print output, only progress bar

    except Exception as e:
        with open(error_log_path, "a") as log_file:
            log_file.write(f"General error: {str(e)}\n")

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Generate medical reports from CT scan folders.")
    parser.add_argument("--excel_file", type=str, required=True, help="Path to the Excel file containing BDMAP IDs.")
    parser.add_argument("--base_folder", type=str, required=True, help="Base directory containing CT and segmentation folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated PDF reports.")
    parser.add_argument("--template_pdf", type=str, required=True, help="Path to the blank PDF template.")
    parser.add_argument("--num_core", type=int, default=0, help="Number of CPU cores to use. Defaults to all available cores.")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
