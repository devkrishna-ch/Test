import requests
import os

if not os.path.exists('Deepfakes'):
    os.makedirs('Deepfakes')


url = 'https://thispersondoesnotexist.com/'


num_images = 100


for i in range(num_images):
    
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

    filename = f'image_{i + 100}.jpg'

    with open(os.path.join('Deepfakes', filename), 'wb') as f:
        f.write(response.content)

    print(f'Downloaded {filename}')

print('Download completed!')
