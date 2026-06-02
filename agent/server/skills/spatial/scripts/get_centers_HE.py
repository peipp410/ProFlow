import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile
import argparse  # For command-line arguments

def get_tile_centers(image, tile_size):
    """
    Generate tile centers for a given image and tile size.
    """
    height, width = image.shape[:2]
    centers = []
    for x in range(0, width, tile_size):
        for y in range(0, height, tile_size):
            center_x = x + tile_size // 2
            center_y = y + tile_size // 2
            centers.append((center_x, center_y))
    return centers

# def segment_tissue_using_hsv(image_bgr):
#     """
#     Segment tissue regions based on purple/pink hues in HSV space.
#     """
#     try:
#         # Convert to HSV space
#         hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

#         # Define HSV ranges for pink and purple tissue
#         lower_purple = np.array([120, 40, 40])  # Hue for purple regions
#         upper_purple = np.array([170, 255, 255])  # Higher range for purple tissue

#         lower_pink = np.array([0, 40, 40])  # Hue for pink regions
#         upper_pink = np.array([15, 255, 255])  # Higher saturation/value for pink

#         # Create masks for purple and pink tissue
#         mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)
#         mask_pink = cv2.inRange(hsv_image, lower_pink, upper_pink)

#         # Combine purple and pink tissue masks
#         combined_mask = cv2.bitwise_or(mask_purple, mask_pink)

#         # Apply morphological operations to clean up the mask
#         kernel = np.ones((5, 5), np.uint8)
#         tissue_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
#         tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)  # Remove noise

#         return tissue_mask
#     except Exception as e:
#         print(f"Error while segmenting tissue using HSV: {e}")
#         return None


def segment_tissue_using_hsv(image_bgr):
    try:
        hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # v = hsv_image[:, :, 2]
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # hsv_image[:, :, 2] = clahe.apply(v)

        lower_purple = np.array([110, 20, 40]) 
        upper_purple = np.array([170, 255, 255])

        lower_pink1 = np.array([0, 15, 40])
        upper_pink1 = np.array([20, 255, 255])
        lower_pink2 = np.array([160, 15, 40])
        upper_pink2 = np.array([179, 255, 255])

        mask_purple = cv2.inRange(hsv_image, lower_purple, upper_purple)
        mask_pink1 = cv2.inRange(hsv_image, lower_pink1, upper_pink1)
        mask_pink2 = cv2.inRange(hsv_image, lower_pink2, upper_pink2)
        combined_mask = cv2.bitwise_or(mask_purple, cv2.bitwise_or(mask_pink1, mask_pink2))

        kernel = np.ones((7, 7), np.uint8)
        tissue_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

        tissue_mask = cv2.GaussianBlur(tissue_mask, (5, 5), 0)

        return tissue_mask

    except Exception as e:
        print(f"Error while segmenting tissue using HSV: {e}")
        return None


def validate_centers_within_boundaries(tissue_mask, tile_centers):
    """
    Validate tile centers by checking if they are inside the tissue boundaries.
    """
    try:
        # Extract contours from tissue mask
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_valid_centers = []

        # Check each tile center against tissue contours
        for center_x, center_y in tile_centers:
            for contour in contours:
                if cv2.pointPolygonTest(contour, (center_x, center_y), measureDist=False) >= 0:  # Inside or on boundary
                    refined_valid_centers.append((center_x, center_y))
                    break  # Move to next center once validated

        return refined_valid_centers
    except Exception as e:
        print(f"Error while validating centers within boundaries: {e}")
        return []

def draw_thumbnail_with_centers(image, valid_centers, thumbnail_path, scale_factor=0.2):
    """
    Draw valid centers on a thumbnail and save it.
    """
    try:
        # Resize image to create thumbnail
        thumbnail = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

        # Draw valid centers on the thumbnail
        for center_x, center_y in valid_centers:
            # Scale the coordinates to match thumbnail dimensions
            x_scaled = int(center_x * scale_factor)
            y_scaled = int(center_y * scale_factor)
            cv2.circle(thumbnail, (x_scaled, y_scaled), 5, (0, 0, 255), -1)  # Red circle on valid centers

        # Save the thumbnail
        cv2.imwrite(thumbnail_path, thumbnail)
        print(f"Thumbnail with valid centers saved to {thumbnail_path}")
    except Exception as e:
        print(f"Error while saving thumbnail: {e}")

def process_single_tiff(file_path, output_dir, tile_size=256, scale_factor=0.2):
    """
    Process a single TIFF file to segment tissue, validate centers, and save outputs.
    """
    try:
        # Load TIFF image
        with tifffile.TiffFile(file_path) as tif:
            image = tif.asarray()

            # Validate image dimensions
            if image.ndim < 2:
                print(f"Skipping file with invalid shape: {file_path}")
                return

            # Convert to BGR format for processing
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Segment tissue using HSV filtering
            tissue_mask = segment_tissue_using_hsv(image_bgr)

            # Generate tile centers
            tile_centers = get_tile_centers(image, tile_size)

            # Validate centers within boundaries defined by contours
            refined_valid_centers = validate_centers_within_boundaries(tissue_mask, tile_centers)

            # Prepare file outputs
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_txt = os.path.join(output_dir, f"valid_centers_{file_name}.txt")
            thumbnail_path = os.path.join(output_dir, f"thumbnail_with_centers_{file_name}.png")

            # Save valid centers to text file
            if refined_valid_centers:
                with open(output_txt, 'w') as f:
                    for center in refined_valid_centers:
                        f.write(f"{center[0]},{center[1]}\n")
                print(f"Valid centers saved to {output_txt}")

            # Draw thumbnail with valid centers
            draw_thumbnail_with_centers(image_bgr, refined_valid_centers, thumbnail_path, scale_factor)

    except Exception as e:
        print(f"Error while processing TIFF file: {e}")

def process_directory(directory, output_dir):
    """
    Process all TIFF files in a directory to identify valid centers and save outputs.
    """
    try:
        # Iterate through TIFF files in the directory
        for file_name in os.listdir(directory):
            if file_name.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(directory, file_name)
                print(f"Processing {file_path}...")
                process_single_tiff(file_path, output_dir, tile_size=50)
    except Exception as e:
        print(f"Error while processing directory: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process TIFF files to segment tissue, validate centers, and save outputs.")
    parser.add_argument("--input_dir", default="/home/peijiazheng/agent/server/skills/spatial/data/images", help="Path to the input directory containing TIFF files.")
    parser.add_argument("--output_dir", default="/home/peijiazheng/agent/server/skills/spatial/data/coords", help="Path to the directory where output files will be saved.")
    parser.add_argument("--tile_size", type=int, default=50, help="Size of tiles to process (default: 50)")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process the directory
    process_directory(args.input_dir, args.output_dir)