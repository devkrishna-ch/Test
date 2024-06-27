import os
from PIL import Image, ImageFile

# Ensure that truncated images can be processed
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_image_valid(image_path):
    """
    Check if an image is valid and can be opened and processed by PIL.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    bool: True if the image is valid, False otherwise.
    """
    try:
        # Attempt to open the image file
        img = Image.open(image_path)
        # Perform a simple operation to ensure the image can be processed
        img.verify()  # Verify if it is a valid image
        img = Image.open(image_path)  # Reopen for further operations
        img.load()  # Load the image to ensure it's not truncated
        return True
    except (OSError, Image.DecompressionBombError) as e:
        print(f"Invalid image {image_path}: {e}")
        return False

def delete_invalid_images(directory):
    """
    Delete truncated, corrupted, or potentially problematic images from a directory.
    
    Args:
    directory (str): Path to the directory containing images.
    """
    # List all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        print(file_path)
        # Check if the file is an image
        if os.path.isfile(file_path) and file_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            # Check if the image is valid
            if not is_image_valid(file_path):
                # If not, delete the image file
                print(f"Deleting invalid image: {file_path}")
                os.remove(file_path)

# Example usage
directory_path = 'RealFaces'  # Replace with the path to your directory
delete_invalid_images(directory_path)
