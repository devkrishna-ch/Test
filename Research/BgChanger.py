import os
from PIL import Image
from rembg import remove

# Set the input and output folders
input_folders = ['deepfakes', 'realfaces']
output_folders = ['Bgfilter-Deepfakes', 'Bgfilter-RealFaces']

# Set the background image file
background_image = Image.open('background.jpg')

# Loop through the input and output folders
for i, input_folder in enumerate(input_folders):
    output_folder = output_folders[i]
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the input image
            input_path = os.path.join(input_folder, filename)
            image = Image.open(input_path)
            
            # Remove the background from the image
            image = remove(image)
            
            # Resize the image to fit the background
            image = image.resize(background_image.size)
            
            # Paste the image onto the background
            background_image = Image.open('background.jpg')
            background_image.paste(image, (0, 0), image)
            
            # Save the output image
            output_path = os.path.join(output_folder, filename)
            background_image.save(output_path)
            print(f'Saved {output_path}')