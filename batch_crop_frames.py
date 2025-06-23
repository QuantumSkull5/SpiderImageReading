import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def align_images(image, reference, max_features=500, keep_percent=0.2):
    """
    Align an image to a reference image using feature matching.
    
    Parameters:
    - image: Image to be aligned
    - reference: Reference image to align to
    - max_features: Maximum number of features to detect
    - keep_percent: Percentage of top matches to keep
    
    Returns:
    - aligned_image: The aligned image
    - homography: The transformation matrix (None if alignment failed)
    """
    
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(max_features)
    (kp1, desc1) = orb.detectAndCompute(image_gray, None)
    (kp2, desc2) = orb.detectAndCompute(reference_gray, None)
    
    # Match features between the two images
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2, None)
    
    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep only the top matches
    num_good_matches = int(len(matches) * keep_percent)
    matches = matches[:num_good_matches]
    
    # Check if we have enough matches
    if len(matches) < 4:
        print("Warning: Not enough matches found for alignment")
        return image, None
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # Find homography
    try:
        (homography, mask) = cv2.findHomography(points1, points2, 
                                              cv2.RANSAC, 4.0)
        
        if homography is None:
            print("Warning: Could not compute homography")
            return image, None
        
        # Apply homography to align the image
        (h, w) = reference.shape[:2]
        aligned = cv2.warpPerspective(image, homography, (w, h))
        
        return aligned, homography
        
    except Exception as e:
        print(f"Warning: Error during alignment: {e}")
        return image, None

def crop_frame_with_alignment(image_path, reference_path=None, output_path=None, 
                             show_result=True, target_size=(2400, 2400), 
                             padding=30, align_images_flag=True):
    """
    Crop scientific frame images with optional alignment to a reference template.
    This function now properly handles cropped reference templates.
    
    Parameters:
    - image_path: Path to input image
    - reference_path: Path to CROPPED reference template
    - output_path: Path to save cropped image (optional)
    - show_result: Whether to display before/after images
    - target_size: Target size for output (width, height)
    - padding: Pixels of padding around detected area
    - align_images_flag: Whether to perform template matching alignment
    
    Returns:
    - cropped_image: PIL Image of the cropped contents
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    original_img = img.copy()
    
    # If we have a reference template, use template matching to find the best crop location
    if reference_path and align_images_flag:
        reference = cv2.imread(reference_path)
        if reference is not None:
            print(f"Finding best match location using template matching...")
            
            # Convert to grayscale for template matching
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # Find the location with highest correlation
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # max_loc is the top-left corner of the best match
            x, y = max_loc
            h, w = template_gray.shape
            
            print(f"✓ Best match found at ({x}, {y}) with confidence {max_val:.3f}")
            
            # Apply padding
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(img.shape[1] - x_padded, w + 2*padding)
            h_padded = min(img.shape[0] - y_padded, h + 2*padding)
            
        else:
            print(f"Warning: Could not load reference template from {reference_path}")
            # Fallback to center crop
            h, w = img.shape[:2]
            crop_size = min(h, w, target_size[0] if target_size else 2400)
            x_padded = (w - crop_size) // 2
            y_padded = (h - crop_size) // 2
            w_padded = h_padded = crop_size
    else:
        # No reference - use center crop
        h, w = img.shape[:2]
        crop_size = min(h, w, target_size[0] if target_size else 2400)
        x_padded = (w - crop_size) // 2
        y_padded = (h - crop_size) // 2
        w_padded = h_padded = crop_size
    
    # Crop the image
    cropped = img[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
    
    # Resize to target size if specified
    if target_size:
        cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to RGB for PIL
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)
    
    # Save if output path provided
    if output_path:
        cropped_pil.save(output_path)
        print(f"Cropped image saved to: {output_path}")
    
    # Display results
    if show_result:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with detected crop area
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image with Detected Crop Area')
        axes[0].axis('off')
        
        # Draw crop rectangle on original image
        rect = plt.Rectangle((x_padded, y_padded), w_padded, h_padded, 
                           linewidth=3, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        # Cropped image
        axes[1].imshow(cropped_rgb)
        axes[1].set_title(f'Cropped Frame ({cropped_rgb.shape[1]}x{cropped_rgb.shape[0]})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return cropped_pil

def batch_crop_with_alignment(input_folder, output_folder, reference_template_path, 
                             target_size=(2400, 2400), padding=30,
                             file_extensions=['jpg', 'jpeg', 'png', 'tiff']):
    """
    Batch crop multiple images using template matching with a cropped reference template.
    
    Parameters:
    - input_folder: Path to folder containing images
    - output_folder: Path to save cropped images
    - reference_template_path: Path to CROPPED reference template
    - target_size: Target size for output (width, height)
    - padding: Pixels of padding for positioning variations
    - file_extensions: List of file extensions to process
    """
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in file_extensions:
        image_files.extend(list(input_path.glob(f'*.{ext}')))
        image_files.extend(list(input_path.glob(f'*.{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to process...")
    print(f"Using reference template: {reference_template_path}")
    
    successful = 0
    
    for i, image_file in enumerate(image_files):
        try:
            output_file = output_path / f"cropped_{image_file.stem}.jpg"
            cropped_img = crop_frame_with_alignment(
                str(image_file), 
                reference_template_path,
                str(output_file), 
                show_result=False,
                target_size=target_size,
                padding=padding,
                align_images_flag=True
            )
            successful += 1
            print(f"✓ Processed {i+1}/{len(image_files)}: {image_file.name}")
        except Exception as e:
            print(f"✗ Error processing {image_file.name}: {str(e)}")
    
    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {successful}/{len(image_files)} images")

def create_reference_template(image_path, output_path, x, y, width, height, show_result=True):
    """
    Create a reference template by cropping to specified coordinates.
    
    Parameters:
    - image_path: Path to input image
    - output_path: Path to save the cropped reference template
    - x: Left edge coordinate of crop area
    - y: Top edge coordinate of crop area  
    - width: Width of crop area
    - height: Height of crop area
    - show_result: Whether to show the cropped reference
    
    Returns:
    - cropped_template: The cropped reference image
    """
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Crop to specified coordinates
    cropped_template = img[y:y+height, x:x+width]
    
    # Save the cropped template
    cv2.imwrite(output_path, cropped_template)
    print(f"Reference template saved to: {output_path}")
    print(f"Template size: {width}x{height}")
    
    if show_result:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with crop area highlighted
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image with Crop Area')
        axes[0].axis('off')
        
        # Draw crop rectangle
        rect = plt.Rectangle((x, y), width, height, 
                           linewidth=3, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        # Cropped template
        cropped_rgb = cv2.cvtColor(cropped_template, cv2.COLOR_BGR2RGB)
        axes[1].imshow(cropped_rgb)
        axes[1].set_title(f'Reference Template ({width}x{height})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return cropped_template
batch_crop_with_alignment(input_folder = r"C:\Files\Araneus diadematus\Missing leg study\Missing leg treatment", output_folder = r"C:\Users\adamain\Downloads\cropped_frames" , reference_template = r"C:\Users\adamain\Downloads\cropped_frames\cropped_EX.jpg", target_size=(2400, 2400), padding=30, file_extensions=['jpg'])