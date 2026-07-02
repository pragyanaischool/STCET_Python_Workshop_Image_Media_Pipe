import streamlit as st
import numpy as np
import mediapipe as mp
import io
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 1. INITIALIZE DASHBOARD CONFIGURATION
st.set_page_config(page_title="PragyanAI Vision Analytics Studio", layout="wide", page_icon="👁️")

# =====================================================================
# 2. MODEL INGESTION LAYER (THREAD-SAFE CACHING ON CPU)
# =====================================================================
@st.cache_resource
def load_vision_pipelines():
    """Downloads and caches model instances onto CPU memory using PIL formatting."""
    yolo_ctx = YOLO("yolov8n.pt")
    
    # Initialize MediaPipe individual component modules
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    mp_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5)
    mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    return yolo_ctx, mp_face, mp_mesh, mp_hands, mp_pose, mp_seg

# Global background engine initialization
yolo, mp_face, mp_mesh, mp_hands, mp_pose, mp_seg = load_vision_pipelines()

# =====================================================================
# 3. INTERACTIVE DASHBOARD RENDERING INTERFACE
# =====================================================================
st.title("PragyanAI Multi-Modal Vision Analytics Studio")
st.caption("A compact production workflow running YOLOv8 tracking alongside MediaPipe topology layers using 100% PIL.")

uploaded_file = st.file_uploader("Ingest target image file workspace...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read data streams directly to standard RGB PIL containers
    orig_img = Image.open(uploaded_file).convert("RGB")
    w, h = orig_img.size
    img_np = np.array(orig_img)

    # Sidebar feature selection toggles
    st.sidebar.header("Pipeline Layer Configurations")
    run_seg = st.sidebar.checkbox("🧼 Apply Background Removal (Segmentation)", value=False)
    run_yolo = st.sidebar.checkbox("🎯 YOLOv8 Object Tracking Bounding Boxes", value=True)
    run_face = st.sidebar.checkbox("👤 MediaPipe Face Boundaries", value=False)
    run_mesh = st.sidebar.checkbox("🕸️ MediaPipe Facial Mesh Coordinates", value=False)
    run_hands = st.sidebar.checkbox("✋ MediaPipe Hand Joint Landmarks", value=False)
    run_pose = st.sidebar.checkbox("🏋️‍♂️ MediaPipe Skeletal Pose Estimation", value=False)

    # Operational Step 1: Execute selfie foreground segmentation using numpy matrix layer mask switches
    if run_seg:
        with st.spinner("Processing segmentation mask contours..."):
            seg_res = mp_seg.process(img_np)
            mask = np.stack((seg_res.segmentation_mask > 0.4,) * 3, axis=-1)
            # Swap background pixel matrices with clean white canvas array layers natively
            img_np = np.where(mask, img_np, np.full(img_np.shape, (255, 255, 255), dtype=np.uint8))

    # Initialize standard PIL mutable canvas layers overrides (Completely avoiding cv2 drawing commands)
    out_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(out_img)
    
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = ImageFont.load_default()

    # Track metrics inside telemetry logs dictionary
    telemetry_summary = {"objects": [], "faces": 0, "hands": 0, "landmarks": 0}

    # Operational Step 2: Execute YOLO Object Boundaries Tracking
    if run_yolo:
        yolo_res = yolo(img_np, verbose=False)[0].boxes
        for box in yolo_res:
            conf = float(box.conf[0])
            if conf > 0.25:
                label = yolo.names[int(box.cls[0])]
                xyxy = box.xyxy[0].tolist()
                telemetry_summary["objects"].append(label)
                
                # Render clean vector box outlines and text via PIL Draw Canvas
                draw.rectangle(xyxy, outline="#00E5FF", width=3)
                draw.text((xyxy[0], max(0, xyxy[1] - 14)), f"{label} {conf:.1%}", fill="#00E5FF", font=font)

    # Operational Step 3: Execute MediaPipe Face Box Tracking
    if run_face:
        face_res = mp_face.process(img_np)
        if face_res.detections:
            telemetry_summary["faces"] = len(face_res.detections)
            for d in face_res.detections:
                b = d.location_data.relative_bounding_box
                xmin, ymin = int(b.xmin * w), int(b.ymin * h)
                xmax, ymax = int((b.xmin + b.width) * w), int((b.ymin + b.height) * h)
                
                draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF00A0", width=3)

    # Operational Step 4: Execute MediaPipe Face Mesh Coordinate Maps
    if run_mesh:
        mesh_res = mp_mesh.process(img_np)
        if mesh_res.multi_face_landmarks:
            for flm in mesh_res.multi_face_landmarks:
                for lm in flm.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill="#00FF00")
                    telemetry_summary["landmarks"] += 1

    # Operational Step 5: Execute MediaPipe Hand Landmark Tracking
    if run_hands:
        hands_res = mp_hands.process(img_np)
        if hands_res.multi_hand_landmarks:
            telemetry_summary["hands"] = len(hands_res.multi_hand_landmarks)
            for hlm in hands_res.multi_hand_landmarks:
                for lm in hlm.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill="#FFFF00")
                    telemetry_summary["landmarks"] += 1

    # Operational Step 6: Execute MediaPipe Skeletal Pose Estimation
    if run_pose:
        pose_res = mp_pose.process(img_np)
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks.landmark:
                if lm.visibility > 0.5:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill="#FF3D00")
                    telemetry_summary["landmarks"] += 1

    # Display results layout splitting panels side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼️ Original Input View")
        st.image(orig_img, use_container_width=True)
    with col2:
        st.markdown("### 🖥️ Deep AI Processing Output")
        st.image(out_img, use_container_width=True)
        
        # Memory-efficient byte stream compilation buffer to handle file download asset
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Annotated Result Image (.png)",
            data=buf.getvalue(),
            file_name="ai_vision_studio_export.png",
            mime="image/png",
            use_container_width=True
        )

    # Telemetry dashboards representation row
    st.divider()
    st.markdown("### Live Analytics Matrix Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Objects Logged", len(telemetry_summary["objects"]))
    kpi2.metric("Faces Verified", telemetry_summary["faces"])
    kpi3.metric("Hands Tracking Profile", telemetry_summary["hands"])
    kpi4.metric("Total Topological Nodes", f"{telemetry_summary['landmarks']} Joints")
