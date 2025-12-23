import streamlit as st
import cv2
import tempfile
import pandas as pd
import numpy as np
import plotly.express as px
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import time
import os
import yt_dlp 


# --- CRITICAL FIX FOR STREAMLIT CLOUD / OPENCV ---
# These specific settings prevent the "CAP_IMAGES" backend
os.environ["OPENCV_VIDEOIO_PRIORITY_CV_IMAGES"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_IMAGES"] = "0"

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Apex Traffic Intelligence Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #FF4B4B; font-weight: 800;}
    .metric-card {
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #FF4B4B; 
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h3 {color: #555555; font-size: 1.2rem; margin-bottom: 5px;}
    .metric-card h2 {color: #333333; font-size: 2rem; font-weight: bold; margin: 0;}
    .stDataFrame {border: 1px solid #ddd; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & DATABASE SETUP
# ==========================================
DB_FILE = "traffic_master_v7.db"

# --- UPDATED: VISDRONE CLASSES ---
# The model.pt was trained on these specific classes, NOT COCO.
VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# Updated color map to match VisDrone keys
COLOR_MAP = {
    'car': '#636EFA', 'van': '#EF553B', 'truck': '#00CC96', 
    'bus': '#AB63FA', 'pedestrian': '#FFA15A', 'people': '#FF6692',
    'bicycle': '#19D3F3', 'motor': '#B6E880', 'tricycle': '#FF97FF', 
    'awning-tricycle': '#FECB52'
}

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS traffic_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            class_name TEXT,
            confidence REAL,
            vehicle_size TEXT,
            lane_position TEXT,
            congestion_status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def clear_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM traffic_logs")
    conn.commit()
    conn.close()

def log_to_db(data_list):
    if not data_list: return
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.executemany('''
        INSERT INTO traffic_logs 
        (timestamp, class_name, confidence, vehicle_size, lane_position, congestion_status) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data_list)
    conn.commit()
    conn.close()

@st.cache_resource
@st.cache_resource
@st.cache_resource
def get_youtube_stream_url(youtube_url):
    # First pass: Get video info to check if it's live
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            is_live = info.get('is_live', False)
    except Exception:
        is_live = False

    # Define options based on live status
    if is_live:
        # For LIVE: Use HLS (m3u8) which is required for continuous streams
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True
        }
    else:
        # For RECORDED: Force a direct HTTP progressive MP4
        # This fixes the "segment loading" error for standard videos
        ydl_opts = {
            'format': 'best[ext=mp4][protocol^=http]',
            'quiet': True,
            'no_warnings': True,
            'geo_bypass': True
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"YouTube Error: {str(e)}")
        return None

init_db()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🎛️ System Controls")

@st.cache_resource
def load_model():
    # Make sure model.pt is in the same directory!
    return YOLO('model.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- INPUT SOURCE SELECTION ---
source = st.sidebar.radio("Video Source", ["Upload Video", "YouTube URL", "Live Camera / RTSP"])
video_path = None

if source == "Upload Video":
    uploaded = st.sidebar.file_uploader("Upload Footage", type=['mp4', 'avi', 'mov'])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        video_path = tfile.name

elif source == "YouTube URL":
    yt_url = st.sidebar.text_input("Paste YouTube Link:", placeholder="https://www.youtube.com/watch?v=...")
    if yt_url:
        with st.spinner("Extracting video stream from YouTube..."):
            stream_url = get_youtube_stream_url(yt_url)
            if stream_url:
                video_path = stream_url
                st.sidebar.success("Stream found!")
            else:
                st.sidebar.error("Could not extract video. Check URL.")

elif source == "Live Camera / RTSP":
    url = st.sidebar.text_input("RTSP URL (or '0' for Webcam)", value="0")
    if url: video_path = 0 if url == "0" else url

# --- FILTERS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Detection Filters")
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35)

use_all_classes = st.sidebar.checkbox(" Select All Classes")
if use_all_classes:
    target_classes = VISDRONE_CLASSES
    st.sidebar.info("Detecting all VisDrone object types.")
else:
    # UPDATED DEFAULT SELECTION TO MATCH VISDRONE
    target_classes = st.sidebar.multiselect(
        "Select Specific Classes", 
        VISDRONE_CLASSES,
        default=['car', 'truck', 'bus', 'pedestrian']
    )

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Database / New Session", type="primary"):
    clear_db()
    st.toast("Database Cleared! Starting fresh.", icon="🧹")
    time.sleep(1)
    st.rerun()

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.markdown('<div class="main-header">Apex Traffic Intelligence Pro</div>', unsafe_allow_html=True)

app_tab, data_tab = st.tabs(["🔴 Live Operations Center", "🗄️ Historical Database"])

# --- TAB 1: LIVE OPERATIONS ---
with app_tab:
    top_col1, top_col2 = st.columns([1.5, 1])
    
    with top_col1:
        st.subheader("📹 Live Surveillance")
        st_frame = st.empty()
        
    with top_col2:
        st.subheader("📊 Real-Time Analytics")
        kpi_holder = st.empty()
        chart_holder_1 = st.empty()
        kpi_holder.info("Waiting for video start...")

    run_btn = st.button("▶️ START SURVEILLANCE", type="primary")
    stop_btn = st.button("⏹️ STOP")

    if run_btn and video_path is not None:
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        
        while cap.isOpened():
            if stop_btn: break
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Detection
            results = model(frame, conf=conf, verbose=False)
            annotated_frame = frame.copy()
            height, width, _ = frame.shape
            
            frame_counts = {cls: 0 for cls in target_classes}
            db_logs = []
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Congestion Logic
            total_objs_in_frame = len(results[0].boxes)
            congestion = "Free Flow"
            if total_objs_in_frame > 10: congestion = "Moderate"
            if total_objs_in_frame > 20: congestion = "Heavy Traffic"

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check bounds to prevent index errors
                if cls_id < len(model.names):
                    cls_name = model.names[cls_id]
                    
                    if cls_name in target_classes:
                        frame_counts[cls_name] = frame_counts.get(cls_name, 0) + 1
                        
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        box_area = (x2-x1) * (y2-y1)
                        
                        v_size = "Medium"
                        if box_area > (width * height * 0.1): v_size = "Large"
                        elif box_area < (width * height * 0.02): v_size = "Small"
                        
                        lane = "Left Lane" if center_x < (width // 2) else "Right Lane"
                        
                        # Fix color logic to handle missing keys nicely
                        color_hex = COLOR_MAP.get(cls_name, '#00FF00')
                        # Convert hex to BGR for OpenCV
                        c_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        color = (c_rgb[2], c_rgb[1], c_rgb[0]) # BGR
                        
                        cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(annotated_frame, f"{cls_name}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        db_logs.append((timestamp_str, cls_name, confidence, v_size, lane, congestion))

            # 2. Log to DB
            if db_logs: log_to_db(db_logs)

            # 3. Update Dashboard (Every 3 frames)
            frame_id += 1
            if frame_id % 3 == 0:
                total_active = sum(frame_counts.values())
                dom_class = max(frame_counts, key=frame_counts.get) if total_active > 0 else "None"
                
                kpi_html = f"""
                <div style="display: flex; gap: 10px;">
                    <div class="metric-card" style="flex: 1;">
                        <h3>Active Objects</h3>
                        <h2>{total_active}</h2>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <h3>Dominant Class</h3>
                        <h2>{dom_class.title()}</h2>
                    </div>
                </div>
                """
                kpi_holder.markdown(kpi_html, unsafe_allow_html=True)
                
                active_counts = {k:v for k,v in frame_counts.items() if v > 0}
                if active_counts:
                    fig_pie = px.pie(
                        names=list(active_counts.keys()), 
                        values=list(active_counts.values()),
                        title="Live Composition",
                        color=list(active_counts.keys()),
                        color_discrete_map=COLOR_MAP,
                        hole=0.4
                    )
                    fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250)
                    chart_holder_1.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{frame_id}")
                else:
                    chart_holder_1.info("No objects detected in current frame.")

            # 4. Display Video
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()

# --- TAB 2: HISTORICAL DATABASE ---
with data_tab:
    st.header("🗄️ Master Database Access")
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM traffic_logs ORDER BY id DESC", conn)
    conn.close()
    
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            f_class = st.multiselect("Filter by Class", df['class_name'].unique(), default=df['class_name'].unique())
        with c2:
            f_lane = st.multiselect("Filter by Lane", df['lane_position'].unique(), default=df['lane_position'].unique())
        with c3:
            f_cong = st.multiselect("Filter by Congestion", df['congestion_status'].unique(), default=df['congestion_status'].unique())
            
        df_filtered = df[
            (df['class_name'].isin(f_class)) & 
            (df['lane_position'].isin(f_lane)) & 
            (df['congestion_status'].isin(f_cong))
        ]
        
        st.markdown("---")
        st.write(f"Showing **{len(df_filtered)}** records matching criteria.")
        
        st.dataframe(
            df_filtered[['timestamp', 'class_name', 'vehicle_size', 'lane_position', 'congestion_status', 'confidence']], 
            use_container_width=True, 
            height=300
        )
        
        st.subheader("📈 Traffic Insights")
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig_lane = px.histogram(df_filtered, x="lane_position", color="class_name", 
                                  title="Lane Usage Distribution", barmode="group",
                                  color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig_lane, use_container_width=True)
            
        with col_b:
            fig_cong = px.pie(df_filtered, names="congestion_status", title="Congestion Levels",
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_cong, use_container_width=True)
            
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Data (CSV)", csv, "traffic_report.csv", "text/csv")
        
    else:
        st.info("Database is empty. Run Live Surveillance to collect data.")