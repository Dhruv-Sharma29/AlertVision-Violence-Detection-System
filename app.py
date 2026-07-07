"""
AlertVision: Violence Detection System

A real-time violence detection application powered by deep learning.
Uses a ResNet50 feature extractor with an LSTM classifier trained on the
RWF-2000 dataset. Provides both file-based video analysis and live webcam
monitoring with email alert capabilities.

Built with Streamlit.
"""

import streamlit as st
import tempfile
import os
import time
import logging
import threading
from datetime import datetime
from io import BytesIO
from collections import deque

import numpy as np
import cv2
from PIL import Image
import smtplib
import ssl
from email.message import EmailMessage
import toml
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

logger = logging.getLogger(__name__)

# ==============================================================================
# Violence Detection Model
# ==============================================================================

@st.cache_resource
def load_lstm_model():
    """Load the trained LSTM model from the models directory."""
    model_path_h5 = os.path.join("models", "best_lstm.h5")
    model_path_keras = os.path.join("models", "best_lstm.keras")

    if os.path.exists(model_path_h5):
        logger.info("Loaded model: models/best_lstm.h5")
        return tf.keras.models.load_model(model_path_h5)
    elif os.path.exists(model_path_keras):
        logger.info("Loaded model: models/best_lstm.keras")
        return tf.keras.models.load_model(model_path_keras)
    else:
        raise FileNotFoundError(
            "No model found in /models folder "
            "(expected best_lstm.h5 or best_lstm.keras)"
        )

@st.cache_resource
def load_feature_extractor():
    """Load the pre-trained ResNet50 feature extractor."""
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load models once at startup
model = load_lstm_model()
feature_extractor = load_feature_extractor()

# Model constants
SEQ_LEN = 16
IMG_SIZE = (224, 224)
FRAME_STRIDE = 10


# ==============================================================================
# Video Analysis
# ==============================================================================

def run_violence_detection(video_path):
    """
    Analyze an uploaded video file for violent content.

    Uses a sliding window approach with frame skipping for performance.
    Extracts ResNet50 features from each sampled frame, then classifies
    sequences of features using the trained LSTM model.

    Args:
        video_path: Path to the video file to analyze.

    Returns:
        A tuple of (prediction, confidence, snapshots) where:
        - prediction: "Violent", "Non-Violent", or "Error"
        - confidence: Float between 0.0 and 1.0
        - snapshots: List of PIL Images captured at detection points
    """
    try:
        feature_queue = deque(maxlen=SEQ_LEN)
        frame_buffer = deque(maxlen=SEQ_LEN)

        snapshots = []
        final_prediction = "Non-Violent"
        max_confidence = 0.0

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_STRIDE != 0:
                frame_count += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(Image.fromarray(rgb_frame))

            resized = cv2.resize(frame, IMG_SIZE)
            preproc = preprocess_input(resized)

            features = feature_extractor.predict(
                np.expand_dims(preproc, axis=0), verbose=0
            )
            feature_queue.append(features[0])

            if len(feature_queue) == SEQ_LEN:
                features_array = np.expand_dims(np.array(feature_queue), axis=0)
                preds = model.predict(features_array, verbose=0)[0]
                confidence = float(preds[0])

                if confidence > st.session_state.violence_threshold:
                    final_prediction = "Violent"
                    middle_frame = frame_buffer[SEQ_LEN // 2]
                    snapshots.append(middle_frame)

                    if confidence > max_confidence:
                        max_confidence = confidence

                elif (1 - confidence) > max_confidence and final_prediction == "Non-Violent":
                    max_confidence = 1 - confidence

            frame_count += 1

        cap.release()

        # Attach timestamps to snapshots for violent detections
        timestamped_snapshots = []
        if final_prediction == "Violent":
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not snapshots:
                fallback_image = Image.new("RGB", (600, 400), color="rgb(42, 43, 46)")
                snapshots = [add_timestamp(fallback_image, timestamp_str, "Snapshot 1 of 1")]
            else:
                if len(snapshots) >= 3:
                    snapshots_to_send = [
                        snapshots[0],
                        snapshots[len(snapshots) // 2],
                        snapshots[-1],
                    ]
                else:
                    snapshots_to_send = snapshots

                for i, snap in enumerate(snapshots_to_send):
                    label = f"Snapshot {i+1} of {len(snapshots_to_send)}"
                    timestamped_snapshots.append(
                        add_timestamp(snap, timestamp_str, label)
                    )

            return final_prediction, max_confidence, timestamped_snapshots

        return final_prediction, max_confidence, []

    except Exception as e:
        logger.error("Error during model inference: %s", e)
        fallback_image = Image.new("RGB", (600, 400), color="rgb(42, 43, 46)")
        return "Error", 0.0, [fallback_image]


# ==============================================================================
# Email Alerts
# ==============================================================================

def load_secrets():
    """Load email credentials from the .streamlit/secrets.toml file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = os.path.join(script_dir, ".streamlit", "secrets.toml")

        if not os.path.exists(secrets_path):
            logger.error("secrets.toml file not found at %s", secrets_path)
            return None

        secrets = toml.load(secrets_path)
        return secrets
    except Exception as e:
        logger.error("Error loading secrets.toml: %s", e)
        return None


def send_alert_email(recipient_list, snapshots):
    """
    Send an email alert with incident snapshots attached.

    Args:
        recipient_list: List of email addresses to notify.
        snapshots: List of PIL Images to attach.

    Returns:
        True if the email was sent successfully, False otherwise.
    """
    secrets = load_secrets()
    if secrets is None or "email_credentials" not in secrets:
        logger.error("Email credentials not found in secrets.toml.")
        return False

    try:
        sender_email = secrets["email_credentials"]["SENDER_EMAIL"]
        sender_password = secrets["email_credentials"]["SENDER_PASSWORD"]

        msg = EmailMessage()
        msg["Subject"] = "ALERT: Violent Activity Detected"
        msg["From"] = sender_email
        msg["To"] = ", ".join(recipient_list)
        msg.set_content(
            "AlertVision has detected a violent incident. "
            "Please see the attached snapshots for details."
        )

        for i, snap in enumerate(snapshots):
            img_buffer = BytesIO()
            snap.save(img_buffer, format="PNG")
            msg.add_attachment(
                img_buffer.getvalue(),
                maintype="image",
                subtype="png",
                filename=f"snapshot_{i+1}.png",
            )

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        logger.info("Alert email sent successfully.")
        return True

    except Exception as e:
        logger.error("Error sending email: %s", e)
        return False


# ==============================================================================
# Image Utilities
# ==============================================================================

def add_timestamp(pil_image, timestamp_str, label_str):
    """
    Burn a timestamp and label onto a PIL image.

    Args:
        pil_image: Source PIL Image (RGB).
        timestamp_str: Timestamp text to overlay at bottom-left.
        label_str: Label text to overlay at top-left.

    Returns:
        A new PIL Image with the overlaid text.
    """
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Timestamp at bottom-left
    (text_width, text_height), baseline = cv2.getTextSize(
        timestamp_str, font, font_scale, font_thickness
    )
    text_x = 10
    text_y = open_cv_image.shape[0] - 10
    cv2.rectangle(
        open_cv_image,
        (text_x, text_y + baseline),
        (text_x + text_width, text_y - text_height - 6),
        bg_color,
        -1,
    )
    cv2.putText(
        open_cv_image, timestamp_str, (text_x, text_y),
        font, font_scale, text_color, 1, cv2.LINE_AA,
    )

    # Label at top-left
    (text_width, text_height), baseline = cv2.getTextSize(
        label_str, font, font_scale, font_thickness
    )
    label_x = 10
    label_y = 10 + text_height
    cv2.rectangle(
        open_cv_image,
        (label_x, label_y + baseline),
        (label_x + text_width, label_y - text_height - 6),
        bg_color,
        -1,
    )
    cv2.putText(
        open_cv_image, label_str, (label_x, label_y),
        font, font_scale, text_color, 1, cv2.LINE_AA,
    )

    return Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))


# ==============================================================================
# Streamlit UI
# ==============================================================================

# Session state initialization
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "prediction" not in st.session_state:
    st.session_state.prediction = ""
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "email_alerts_enabled" not in st.session_state:
    st.session_state.email_alerts_enabled = True
if "alert_emails" not in st.session_state:
    st.session_state.alert_emails = ""
if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False
if "stop_webcam" not in st.session_state:
    st.session_state.stop_webcam = False
if "violence_threshold" not in st.session_state:
    st.session_state.violence_threshold = 0.5

# Page configuration
st.set_page_config(page_title="AlertVision", layout="wide")

st.markdown("""
<style>
[data-testid="stFileUploaderInstructions"] p {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 20px;'>"
    "AlertVision: Violence Detection System</h1>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Real-time Analysis", "Configuration"])

st.sidebar.title("About")
st.sidebar.info(
    "AlertVision is a deep learning-based violence detection system. "
    "Upload a video for analysis or use your webcam for real-time monitoring. "
    "The system uses ResNet50 + LSTM trained on the RWF-2000 dataset."
)


# ==============================================================================
# Page: Real-time Analysis
# ==============================================================================

if page == "Real-time Analysis":

    input_source = st.radio(
        "Select Input Source:",
        ("Upload Video File", "Use Live Webcam"),
        horizontal=True,
    )

    st.header("Video Input & Analysis")

    # --------------------------------------------------------------------------
    # Upload Video File
    # --------------------------------------------------------------------------
    if input_source == "Upload Video File":
        st.session_state.stop_webcam = True

        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=["mp4", "avi", "mov", "mkv", "mpeg4"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{uploaded_file.name.split('.')[-1]}",
                ) as tfile:
                    tfile.write(uploaded_file.read())
                    st.session_state.temp_video_path = tfile.name
                    st.session_state.uploaded_file_name = uploaded_file.name

                st.session_state.analysis_done = False
                st.session_state.snapshots = []
                st.session_state.alert_sent = False

        if st.session_state.temp_video_path is not None:
            vid_col_1, vid_col_2, vid_col_3 = st.columns([0.35, 0.3, 0.35])
            with vid_col_2:
                st.video(st.session_state.temp_video_path)

        st.divider()
        st.subheader("Run Analysis")
        analyze_button = st.button(
            "Analyze Video", type="primary", use_container_width=True
        )
        results_container = st.container()

        if analyze_button:
            if st.session_state.temp_video_path is not None:
                with st.spinner("Processing video..."):
                    st.session_state.analysis_done = False
                    st.session_state.snapshots = []
                    st.session_state.alert_sent = False

                    prediction, confidence, snapshots = run_violence_detection(
                        st.session_state.temp_video_path
                    )
                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.session_state.snapshots = snapshots
                    st.session_state.analysis_done = True
            else:
                st.warning("Please upload a video file first.")

        if st.session_state.analysis_done:
            with results_container:
                st.subheader("Analysis Results")
                confidence = st.session_state.get("confidence", 0.0)

                if st.session_state.prediction == "Violent":
                    st.markdown(f"""
                    <div style="
                        background-color: #E67E22;
                        border: 1px solid #D35400;
                        border-radius: 5px;
                        padding: 10px;
                        color: white;
                        font-weight: bold;
                        font-family: 'Source Sans Pro', sans-serif;
                    ">
                        Result: VIOLENCE DETECTED ({confidence*100:.2f}% confidence)
                    </div>
                    """, unsafe_allow_html=True)

                    if (
                        st.session_state.email_alerts_enabled
                        and st.session_state.alert_emails
                    ):
                        email_list = [
                            email.strip()
                            for email in st.session_state.alert_emails.split(",")
                            if email.strip()
                        ]
                        if email_list:
                            if not st.session_state.alert_sent:
                                with st.spinner("Sending email alerts..."):
                                    success = send_alert_email(
                                        email_list, st.session_state.snapshots
                                    )
                                    if success:
                                        st.session_state.alert_sent = True
                                        st.success(
                                            f"Alert successfully sent to: "
                                            f"**{', '.join(email_list)}**"
                                        )
                                    else:
                                        st.error(
                                            "Failed to send email alert. "
                                            "Check credentials in secrets.toml."
                                        )
                            elif st.session_state.alert_sent:
                                st.success(
                                    f"Alert successfully sent to: "
                                    f"**{', '.join(email_list)}**"
                                )

                elif st.session_state.prediction == "Non-Violent":
                    st.success(
                        f"**Result: Non-Violent ({confidence*100:.2f}% confidence)**"
                    )

                else:
                    st.error(f"**Result: {st.session_state.prediction}**")

        if st.session_state.analysis_done and st.session_state.snapshots:
            st.divider()
            st.header("Detected Incident Snapshots")
            snapshot_cols = st.columns(3)
            for i, snap in enumerate(st.session_state.snapshots):
                with snapshot_cols[i % 3]:
                    st.image(snap, caption=f"Snapshot {i+1}", use_container_width=True)

    # --------------------------------------------------------------------------
    # Live Webcam
    # --------------------------------------------------------------------------
    elif input_source == "Use Live Webcam":
        st.info("Click 'Start Webcam' to begin real-time analysis.")

        start_button = st.button("Start Webcam", key="start_cam")

        stop_placeholder = st.empty()
        frame_placeholder = st.empty()
        alert_status_placeholder = st.empty()

        if start_button:
            st.session_state.stop_webcam = False

            feature_queue = deque(maxlen=SEQ_LEN)
            frame_buffer = deque(maxlen=SEQ_LEN)
            last_alert_time = 0
            alert_cooldown_seconds = 30

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Could not open webcam. Please check permissions.")
                st.stop()

            with stop_placeholder.container():
                if st.button("Stop Webcam", key="stop_cam"):
                    st.session_state.stop_webcam = True

            while cap.isOpened() and not st.session_state.stop_webcam:
                ret, frame_np = cap.read()
                if not ret:
                    st.warning("Webcam feed lost.")
                    break

                rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                frame_buffer.append(Image.fromarray(rgb_frame))
                resized = cv2.resize(frame_np, IMG_SIZE)
                preproc = preprocess_input(resized)
                features = feature_extractor.predict(
                    np.expand_dims(preproc, axis=0), verbose=0
                )
                feature_queue.append(features[0])

                prediction_text = "Processing..."
                prediction_confidence = 0.0

                if len(feature_queue) == SEQ_LEN:
                    features_array = np.expand_dims(
                        np.array(feature_queue), axis=0
                    )
                    preds = model.predict(features_array, verbose=0)[0]
                    confidence = float(preds[0])

                    if confidence > st.session_state.violence_threshold:
                        prediction_text = "VIOLENCE DETECTED"
                        prediction_confidence = confidence

                        current_time = time.time()
                        if (current_time - last_alert_time) > alert_cooldown_seconds:
                            last_alert_time = current_time

                            timestamp_str = datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )[:-3]

                            snap_pil_1 = frame_buffer[0]
                            snap_pil_2 = frame_buffer[SEQ_LEN // 2]
                            snap_pil_3 = frame_buffer[SEQ_LEN - 1]

                            snap_ts_1 = add_timestamp(
                                snap_pil_1, timestamp_str, "Snapshot 1 of 3"
                            )
                            snap_ts_2 = add_timestamp(
                                snap_pil_2, timestamp_str, "Snapshot 2 of 3"
                            )
                            snap_ts_3 = add_timestamp(
                                snap_pil_3, timestamp_str, "Snapshot 3 of 3"
                            )

                            snapshots_list = [snap_ts_1, snap_ts_2, snap_ts_3]

                            if (
                                st.session_state.email_alerts_enabled
                                and st.session_state.alert_emails
                            ):
                                email_list = [
                                    email.strip()
                                    for email in st.session_state.alert_emails.split(",")
                                    if email.strip()
                                ]
                                if email_list:
                                    alert_thread = threading.Thread(
                                        target=send_alert_email,
                                        args=(email_list, snapshots_list),
                                    )
                                    alert_thread.start()
                                    logger.info("Alert email thread started.")

                                    alert_status_placeholder.success(
                                        f"Alert sent to: "
                                        f"**{', '.join(email_list)}**"
                                    )
                    else:
                        prediction_text = "NORMAL"
                        prediction_confidence = 1 - confidence
                        alert_status_placeholder.empty()

                # Overlay prediction on frame
                color = (
                    (0, 0, 255)
                    if prediction_text == "VIOLENCE DETECTED"
                    else (0, 255, 0)
                )
                text = f"{prediction_text} ({prediction_confidence*100:.1f}%)"

                live_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    frame_np, live_timestamp_str,
                    (10, frame_np.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame_np, live_timestamp_str,
                    (10, frame_np.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
                )

                cv2.putText(
                    frame_np, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA,
                )
                if prediction_text == "VIOLENCE DETECTED":
                    cv2.rectangle(
                        frame_np, (0, 0),
                        (frame_np.shape[1], frame_np.shape[0]),
                        color, 5,
                    )

                frame_placeholder.image(
                    cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB), channels="RGB"
                )

            cap.release()
            cv2.destroyAllWindows()
            if st.session_state.stop_webcam:
                st.warning("Webcam stopped by user.")

# ==============================================================================
# Page: Configuration
# ==============================================================================

elif page == "Configuration":
    st.header("System Configuration")
    st.divider()
    st.subheader("Alerting Options")
    st.session_state.email_alerts_enabled = st.toggle(
        "Enable Email Alerts",
        value=st.session_state.email_alerts_enabled,
    )

    if st.session_state.email_alerts_enabled:
        st.session_state.alert_emails = st.text_area(
            "Enter Email(s) for Alerts (comma-separated):",
            value=st.session_state.alert_emails,
        )

    st.divider()
    st.subheader("Model Sensitivity")
    st.session_state.violence_threshold = st.slider(
        "Violence Detection Threshold (Default: 0.5)",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.violence_threshold,
        step=0.05,
        help=(
            "Lower values are MORE sensitive (more alerts). "
            "Higher values are LESS sensitive (fewer alerts)."
        ),
    )

    st.divider()
    st.info("Your configuration is saved automatically for this session.")