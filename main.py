import streamlit as st
from PIL import Image
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from deepfake_detector import predict, calculate_metrics
from email_Feedback import send_email
from SaveUploadedImg import save_uploaded_file

def main():
    # Initialize session state
    session_state = st.session_state
    if 'uploaded_count' not in session_state:
        session_state.uploaded_count = 0
    if 'real_count' not in session_state:
        session_state.real_count = 0
    if 'fake_count' not in session_state:
        session_state.fake_count = 0
    if 'ai_generated_count' not in session_state:
        session_state.ai_generated_count = 0
    if 'recent_uploads' not in session_state:
        session_state.recent_uploads = []

    st.title('Deepfake Detection App')
    st.sidebar.image("colored-logo.jpg", use_column_width=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Deepfake Detection", "About Us", "Dashboard"])

    if page == "Home":
        home()
    elif page == "About Us":
        about_us()
    elif page == "Deepfake Detection":
        deepfake()
    elif page == "Dashboard":
        dashboard(session_state)

def about_us():
    st.title("About Us")

    st.write("""
    Welcome to our AI-powered application! We are a team of passionate developers dedicated to creating innovative solutions.
    
    Our product utilizes cutting-edge artificial intelligence technology to streamline processes and provide valuable insights.

    **Key Features**:
    - Advanced machine learning algorithms
    - Intuitive user interface
    - Real-time data analysis         
    
    """)

    # Add columns for team members and contact information
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Team Members")
        st.write("""
        - **Raman Puri**: Team Lead
        - **Devansh**: AI Researcher
        - **Krishna**: Developer
        - **Aryan**: AI Engineer
        """)

    with col2:
        st.subheader("Contact Us")
        st.write("""
        For more information, visit our [website](https://svam.com) or reach out to us via email at [dverma@svam.com](mailto:dverma@svam.com).
        """)

    # Add an image
    st.image("back.png", caption=None, use_column_width=True)

    # Add a button for feedback
    st.write("**Give Feedback**")

    # Initialize session state for feedback type
    if 'feedback_type' not in st.session_state:
        st.session_state.feedback_type = None

    # Create buttons for feedback type
    if st.button('Like'):
        st.session_state.feedback_type = 'Like'
    elif st.button('Dislike'):
        st.session_state.feedback_type = 'Dislike'

    # Display current feedback type
    if st.session_state.feedback_type:
        st.write(f'You selected: {st.session_state.feedback_type}')

    # Text area for feedback
    seder_name=st.text_input("Enter your name:")
    seder_email=st.text_input("Enter your email:")
    sender_phone=st.text_input("Enter your phone number:")
    feedback = st.text_area("Please leave your feedback here:")

    uploaded_Img = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    img_path = None
    if uploaded_Img is not None:
        img_path = save_uploaded_file(uploaded_Img)
    if img_path is not None:
        st.write("Image saved to:", img_path)

    # Submit button
    if st.button('Submit'):
        if feedback:
            if st.session_state.feedback_type:
                # Replace the following line with your email sending logic
                send_email(st.session_state.feedback_type, feedback,seder_name,seder_email,sender_phone,img_path)
                st.success('Thank you for your feedback! It has been submitted successfully.')
                # Clear the feedback type after submission
                st.session_state.feedback_type = None
            else:
                st.warning('Please select either "Like" or "Dislike" before submitting.')
        else:
            st.warning('Please provide feedback before submitting.')

def home():
    header_image = "deepfake_detection_header.jpg"
    st.image(header_image, use_column_width=True)
    st.write("""
    Welcome to our Deepfake Detection App! This application utilizes advanced machine learning algorithms to detect and identify deepfake images.
    
    **How it Works**:
    1. Upload an image containing a suspected deepfake.
    2. Our AI model will analyze the content and provide a prediction on whether it is real, deepfake, or AI generated.
    3. You'll receive the prediction result along with a confidence score.    
    
    **Key Features**:
    - Upload images for analysis
    - Real-time deepfake detection
    - Confidence score for each prediction
    
    **Disclaimer**: While our AI model is highly accurate, please note that no detection system is perfect. Always use your judgment when assessing the authenticity of media content.
    
    Ready to get started? Upload your file and let's detect some deepfakes!""")

def deepfake():
    session_state = st.session_state
    global uploaded_count, real_count, fake_count, ai_generated_count

    # Uploading file
    st.sidebar.info("Please select the file type for upload from the dropdown menu.")
    st.sidebar.title("Upload Options")
    upload_option = st.sidebar.selectbox("Choose the upload type:", ["Single Image Upload", "Batch Upload"])

    if upload_option == "Single Image Upload":
        st.header("Single Image Upload")
        st.write("**Instructions:**")
        st.write("- **Click on Browse.**")
        st.write("- **Upload a single image file (jpg, jpeg, or png) to check if it is real, fake, or AI generated.**")
        st.write("- **The application will display the uploaded image along with prediction results and a GradCAM visualization.**")
        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            session_state.uploaded_count += 1
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            try:
                confidences, prediction, gradcam_image = predict(image)
                if prediction == "real":
                    session_state.real_count += 1
                elif prediction == "fake":
                    session_state.fake_count += 1
                else:
                    session_state.ai_generated_count += 1

                st.subheader("Prediction:")
                # st.write("Single uploaded image count:", session_state.uploaded_count)
                # st.write("Real count:", session_state.real_count)
                # st.write("Fake count:", session_state.fake_count)
                # st.write("AI Generated count:", session_state.ai_generated_count)
                st.write("The uploaded image is classified as:", prediction)
                st.write("Confidence Scores:")
                st.write(confidences)
                st.image(gradcam_image, caption="GradCAM", use_column_width=True)
            except Exception as e:
                st.error("Error predicting the image:")
                st.error(str(e))

    elif upload_option == "Batch Upload":
        batch_option = st.selectbox("Choose the batch upload type:", ["Multiple Files", "Entire Folder"])

        if batch_option == "Multiple Files":
            st.write("**Instructions:**")
            st.write("- **Upload multiple image files (jpg, jpeg, or png).**")
            st.write("- **Select the true label for each image.**")
            st.write("- **The application will display predictions and GradCAM visualizations for each image and other metrics.**")
            uploaded_files = st.file_uploader("Upload images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

            if uploaded_files:
                true_labels = []
                for uploaded_file in uploaded_files:
                    true_label = st.selectbox(f"Select the true label for {uploaded_file.name}", ["real", "fake", "ai_generated"], key=uploaded_file.name)
                    true_labels.append(true_label)

                if st.button("Start Batch Prediction"):
                    predictions = []
                    gradcam_images = []

                    for idx, uploaded_file in enumerate(uploaded_files):
                        session_state.uploaded_count += 1
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f"Uploaded Image {idx + 1}", use_column_width=True)

                        try:
                            confidences, prediction, gradcam_image = predict(image)
                            if prediction == "real":
                                session_state.real_count += 1
                            elif prediction == "fake":
                                session_state.fake_count += 1
                            else:
                                session_state.ai_generated_count += 1

                            predictions.append(prediction)
                            gradcam_images.append(gradcam_image)

                            if prediction == true_labels[idx]:
                                st.success(f"Prediction: {prediction} (Correct)")
                            else:
                                st.error(f"Prediction: {prediction} (Incorrect)")

                            st.image(gradcam_image, caption=f"GradCAM {idx + 1}", use_column_width=True)
                        except Exception as e:
                            st.error("Error predicting the image:")
                            st.error(str(e))

                    st.subheader("Metrics:")
                    # st.write("Batch uploaded image count:", session_state.uploaded_count)
                    # st.write("Real count:", session_state.real_count)
                    # st.write("Fake count:", session_state.fake_count)
                    # st.write("AI Generated count:", session_state.ai_generated_count)
                    metrics = calculate_metrics(true_labels, predictions)
                    st.write(f"Accuracy: {metrics['accuracy']}")
                    st.write(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
                    st.write(f"F1 Score: {metrics['f1_score']}")
                    st.write(f"Precision: {metrics['precision']}")
                    st.write(f"Recall: {metrics['recall']}")

        elif batch_option == "Entire Folder":
            st.write("**Instructions:**")
            st.write("- **Enter the path of the folder containing the images.**")
            st.write("- **Upload a CSV file with the image names and their corresponding true labels.**")
            st.write("- **The application will display predictions and GradCAM visualizations for each image and other metrics.**")

            folder_path = st.text_input("Enter the path of the folder containing the images:")
            csv_file = st.file_uploader("Upload CSV file with image names and true labels", type=["csv"])

            if st.button("Start Batch Prediction"):
                if folder_path and csv_file:
                    # Load labels from CSV file
                    labels_df = pd.read_csv(csv_file)
                    labels_dict = dict(zip(labels_df['image_name'], labels_df['true_label']))

                    all_images = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('jpg', 'jpeg', 'png'))]

                    predictions = []
                    gradcam_images = []
                    true_labels = []

                    for idx, image_path in enumerate(all_images):
                        image_name = os.path.basename(image_path)
                        if image_name in labels_dict:
                            true_label = labels_dict[image_name]
                            true_labels.append(true_label)

                            session_state.uploaded_count += 1
                            image = Image.open(image_path)
                            st.image(image, caption=f"Uploaded Image {idx + 1}", use_column_width=True)

                            try:
                                confidences, prediction, gradcam_image = predict(image)
                                if prediction == "real":
                                    session_state.real_count += 1
                                elif prediction == "fake":
                                    session_state.fake_count += 1
                                else:
                                    session_state.ai_generated_count += 1

                                predictions.append(prediction)
                                gradcam_images.append(gradcam_image)

                                if prediction == true_label:
                                    st.success(f"Prediction: {prediction} (Correct)")
                                else:
                                    st.error(f"Prediction: {prediction} (Incorrect)")

                                st.image(gradcam_image, caption=f"GradCAM {idx + 1}", use_column_width=True)
                            except Exception as e:
                                st.error("Error predicting the image:")
                                st.error(str(e))

                    st.subheader("Metrics:")
                    # st.write("Folder uploaded image count:", session_state.uploaded_count)
                    # st.write("Real count:", session_state.real_count)
                    # st.write("Fake count:", session_state.fake_count)
                    # st.write("AI Generated count:", session_state.ai_generated_count)
                    metrics = calculate_metrics(true_labels, predictions)
                    st.write(f"Accuracy: {metrics['accuracy']}")
                    st.write(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
                    st.write(f"F1 Score: {metrics['f1_score']}")
                    st.write(f"Precision: {metrics['precision']}")
                    st.write(f"Recall: {metrics['recall']}")

def dashboard(session_state):
    st.subheader("Analytics")

    # Create a DataFrame for visualization
    data = {
        "Category": ["Total Uploaded Images", "Real Images", "Fake Images", "AI Generated Images"],
        "Count": [session_state.uploaded_count, session_state.real_count, session_state.fake_count, session_state.ai_generated_count]
    }
    df = pd.DataFrame(data)
    st.write(df)

    # Plot counts
    st.write("### Counts Overview")
    fig, ax = plt.subplots()
    custom_palette = ["#337AFF", "#33FF57", "#FF5733", "#FF33FF"]
    sns.barplot(x="Category", y="Count", data=df, ax=ax, palette=custom_palette)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Display recent uploads
    st.write("### Recent Uploads")
    if session_state.recent_uploads:
        for upload in session_state.recent_uploads:
            st.image(upload["image"], caption=f"Uploaded Image ({upload['type']})", use_column_width=True)
            st.write("Prediction:", upload["prediction"])
            st.write("Confidence Score:", upload["confidence"])
    else:
        st.write("No recent uploads.")

if __name__ == "__main__":
    main()
