import io
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import os

# Define the necessary variables
SERVICE_ACCOUNT_FILE = 'cred.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FILE_ID = '1KxaY7cFO_ie6KMq1uf8vu68wTc8x6hhT'
DESTINATION_FILE_PATH = 'Weights&Labels/resnetinceptionv1_epoch.pth'

def download_file(service_account_file=SERVICE_ACCOUNT_FILE, scopes=SCOPES, file_id=FILE_ID, destination_file_path=DESTINATION_FILE_PATH):
    # Authenticate with the Google Drive API using the service account credentials
    file_path = DESTINATION_FILE_PATH
    if os.path.exists(file_path):
        print("File exists.")
    else:
        print("File does not exist, start downloding...")
        creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes)

        # Build the Google Drive API service
        service = build('drive', 'v3', credentials=creds)

        # Request to download the file
        request = service.files().get_media(fileId=file_id)

        # Download the file
        with io.FileIO(destination_file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")

        print("File downloaded successfully.")

if __name__ == '__main__':
   download_file(SERVICE_ACCOUNT_FILE,SCOPES, FILE_ID, DESTINATION_FILE_PATH)