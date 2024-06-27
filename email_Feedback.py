import streamlit as st
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_email(feedback_type,feedback,Name,Email,Phone,img_path):
    # Set up the SMTP server
    smtp_server = 'smtp.gmail.com'
    port = 587  # or another port
    sender_email = 'vjtsvam@gmail.com'
    receiver_email = 'krishna.dev@svam.com;dverma@svam.com'
    password = 'lfjl peoi eifz lybz'

    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f'Testing - Deepfake Detection App | User Feedback - {feedback_type}'

    # Attach the image to the email
    if img_path is not None:
        with open(img_path, 'rb') as fp:
            img_data = fp.read()
            img = MIMEImage(img_data)
            msg.attach(img)

    # Add message body
    body = (f" Hi Team, \n\n Please find the feedback and user details below :\n\n"
        f"Feedback Types:  {feedback_type}\n"
        f"Feedback from user:  {feedback}\n\n"
        f"Sender Name:  {Name}\n"
        f"Sender Email:  {Email}\n"
        f"Sender Phone:  {Phone}\n\n"
        "Note - Also, refer to the image for reference if attached. \n\n Thanks, \n SVAM International Inc.")
    msg.attach(MIMEText(body, 'plain'))

    # Connect to SMTP server and send email
    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        os.remove(img_path)
    except Exception as e:
        print(f"Failed to send email: {e}")