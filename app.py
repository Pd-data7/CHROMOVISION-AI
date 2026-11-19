import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
from PIL import Image

from modules.preprocessing import apply_denoising, calculate_noise_ratio
from modules.color_detection import detect_colors, COLOR_RANGES
from modules.contour_detection import analyze_contours
from modules.analytics import analyze_detection_data, benchmark_processing, export_detection_report


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CHROMOVISION AI | Vision Intelligence",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DESIGN SYSTEM
# =========================================================
COLOR_HEX = {
    "Red": "#ff5470", "Orange": "#ff9f43", "Yellow": "#ffd166",
    "Green": "#2dd4a7", "Cyan": "#35d5f2", "Blue": "#4d9fff",
    "Purple": "#a78bfa", "Pink": "#ff78bd", "White": "#e8edf5",
    "Black": "#8792a5"
}
COLOR_EMOJI = {
    "Red": "🔴", "Orange": "🟠", "Yellow": "🟡", "Green": "🟢",
    "Cyan": "🩵", "Blue": "🔵", "Purple": "🟣", "Pink": "🩷",
    "White": "⚪", "Black": "⚫"
}
PLOT_TEMPLATE = "plotly_dark"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #080d1a;
    --panel: #111a2d;
    --panel2: #15223a;
    --line: rgba(126, 167, 226, .17);
    --muted: #9aabc4;
    --cyan: #45d8f2;
    --violet: #9b8cff;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(ellipse at 8% 0%, rgba(20, 184, 220, .13), transparent 32%),
      radial-gradient(ellipse at 95% 8%, rgba(130, 91, 255, .14), transparent 30%),
      linear-gradient(145deg, #080d1a 0%, #0b1324 55%, #0b1020 100%);
    color: #eef4ff;
}

/* FIX: more top padding so Streamlit's fixed top bar no longer hides the title */
.block-container { max-width: 1500px; padding-top: 4.5rem; padding-bottom: 3rem; }

/* FIX: make Streamlit's top bar blend with the dark theme */
header[data-testid="stHeader"] {
    background: rgba(8, 13, 26, .75);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
}

/* FIX: give the sidebar a little breathing room at the top */
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.45px; }
h1 { color: #f7fbff !important; font-size: 2.45rem !important; font-weight: 700 !important; }
h2 { color: #eaf4ff !important; font-size: 1.55rem !important; }
h3 { color: #70def2 !important; font-size: 1.12rem !important; }
p, label, .stCaption { color: #b7c5da; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111d33 0%, #0a1222 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h1 { font-size: 1.45rem !important; color: #6be4f6 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebdd3; }
div[role="radiogroup"] label {
    background: rgba(255,255,255,.025);
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 8px 11px;
    margin: 3px 0;
    transition: .2s ease;
}
div[role="radiogroup"] label:hover {
    background: rgba(69,216,242,.09);
    border-color: rgba(69,216,242,.25);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(24,39,65,.96), rgba(15,25,44,.96));
    border: 1px solid rgba(100,164,226,.20);
    border-radius: 16px;
    padding: 17px 18px;
    box-shadow: 0 9px 26px rgba(0,0,0,.16);
}
[data-testid="stMetricLabel"] { color: #9fb3cf !important; }
[data-testid="stMetricValue"] { color: #65ddf2 !important; font-family: 'Space Grotesk', sans-serif; }
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(100deg, #087fa7, #6556ce);
    color: white !important;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 11px;
    padding: .58rem 1rem;
    font-weight: 700;
    transition: .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(100deg, #0aaed0, #8a6dff);
    border-color: rgba(255,255,255,.28);
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(25,177,220,.18);
}
[data-testid="stFileUploader"] {
    background: rgba(26,43,69,.55);
    border: 1px dashed rgba(69,216,242,.55);
    border-radius: 14px;
    padding: 10px;
}
div[data-baseweb="select"] > div {
    background: #14223a;
    border: 1px solid #304966;
    border-radius: 10px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stImage"] img { border-radius: 13px; border: 1px solid var(--line); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.035);
    border-radius: 9px 9px 0 0;
    padding: 9px 15px;
}
.stTabs [aria-selected="true"] { color: #66e1f4 !important; }
hr { border-color: rgba(126,167,226,.16); }
[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stExpander"] {
    background: rgba(18,31,52,.65);
    border: 1px solid var(--line);
    border-radius: 12px;
}
.hero {
    background:
      radial-gradient(circle at 82% 15%, rgba(155,140,255,.25), transparent 27%),
      radial-gradient(circle at 15% 90%, rgba(53,213,242,.18), transparent 30%),
      linear-gradient(115deg, #142642, #171d3a 65%, #202044);
    border: 1px solid rgba(120,176,240,.25);
    border-radius: 22px;
    padding: 28px 30px;
    margin: 5px 0 22px 0;
    box-shadow: 0 18px 45px rgba(0,0,0,.18);
}
.hero-kicker { color: #68e0f3; font-size: .78rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.hero-title { color: #fff; font-family: 'Space Grotesk',sans-serif; font-size: 2.1rem; font-weight: 700; margin: 7px 0; }
.hero-copy { color: #c4d2e7; font-size: 1rem; max-width: 780px; line-height: 1.65; }
.section-kicker { color: #59d9ed; text-transform: uppercase; letter-spacing: 1.7px; font-size: .72rem; font-weight: 700; margin-bottom: 4px; }
.colour-tile {
    background: linear-gradient(145deg, rgba(25,41,67,.95), rgba(14,24,42,.95));
    border: 1px solid rgba(130,165,210,.16);
    border-radius: 13px;
    padding: 13px 14px;
    margin-bottom: 9px;
}
.colour-dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:8px; }
.small-muted { color:#91a4bf; font-size:.85rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def section_heading(kicker, title, description=None):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if description:
        st.caption(description)


def show_image_pair(left_img, right_img, left_title, right_title, left_is_bgr=False, right_is_bgr=False):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**{left_title}**")
        if left_is_bgr:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        st.image(left_img, use_container_width=True)
    with c2:
        st.markdown(f"**{right_title}**")
        if right_is_bgr:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        st.image(right_img, use_container_width=True)


def plot_layout(fig, height=360):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce8f8", family="DM Sans"),
        title_font=dict(size=17, color="#f1f6ff"),
        margin=dict(l=15, r=15, t=55, b=15),
        height=height,
        legend_title_text=""
    )
    fig.update_xaxes(gridcolor="rgba(150,180,220,.10)", zerolinecolor="rgba(150,180,220,.10)")
    fig.update_yaxes(gridcolor="rgba(150,180,220,.10)", zerolinecolor="rgba(150,180,220,.10)")
    return fig


def get_detection(image, area):
    return detect_colors(image, min_area=area)


def get_rgb_from_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# =========================================================
# SESSION STATE
# =========================================================
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

if "last_detection" not in st.session_state:
    st.session_state.last_detection = None


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🎨 CHROMOVISION")
    st.caption("VISION INTELLIGENCE PLATFORM")
    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview", "Image Detection", "Segmentation",
            "HSV Calibration", "Noise Analysis", "Lighting Experiments",
            "Performance", "Analytics", "Detection History"
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### 📤 Image Workspace")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG."
    )

    min_area = st.slider(
        "Minimum object area",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        help="Increase this value to ignore smaller detected regions."
    )

    st.markdown("---")
    st.markdown("**SYSTEM MODULES**")
    st.caption("OpenCV · HSV segmentation · Contours · Plotly analytics")
    st.caption("Prototype build | Uploaded-image analysis")


# =========================================================
# LOAD IMAGE SAFELY
# =========================================================
rgb_image = None
bgr_image = None
if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        rgb_image = np.array(pil_image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")


# =========================================================
# TOP HEADER
# =========================================================
top_left, top_right = st.columns([5, 1])
with top_left:
    st.markdown("### 🎨 CHROMOVISION AI")
    st.caption("INTELLIGENT MULTI-COLOUR OBJECT DETECTION & IMAGE ANALYSIS")
with top_right:
    st.markdown(
        '<div style="text-align:right;color:#6de2f3;font-weight:700;padding-top:12px;">● SYSTEM READY</div>',
        unsafe_allow_html=True
    )
st.markdown("---")


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">Computer Vision • Image Processing • Data Analytics</div>
      <div class="hero-title">See colour. Detect objects. Understand images.</div>
      <div class="hero-copy">
        CHROMOVISION AI transforms an uploaded image into measurable visual information
        using HSV colour thresholding, binary segmentation, morphological filtering
        and contour-based object analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Colour classes", len(COLOR_RANGES), "HSV threshold groups")
    m2.metric("Core engine", "OpenCV", "Image processing")
    m3.metric("Interface", "Streamlit", "Interactive dashboard")
    m4.metric("Analytics", "Plotly", "Visual reports")

    st.markdown("")
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        section_heading("01 / PIPELINE", "How the system works")
        steps = [
            ("01", "Image input", "Upload a JPG or PNG image."),
            ("02", "Preprocessing", "Prepare the image for stable analysis."),
            ("03", "BGR → HSV", "Convert colour representation for thresholding."),
            ("04", "Colour segmentation", "Create binary masks for supported colour groups."),
            ("05", "Morphology + contours", "Clean masks and locate candidate regions."),
            ("06", "Object analytics", "Measure area, perimeter and bounding geometry.")
        ]
        for number, title, desc in steps:
            st.markdown(
                f"""<div class="colour-tile">
                <span style="color:#59d9ed;font-weight:700;margin-right:12px">{number}</span>
                <b style="color:#edf5ff">{title}</b><br>
                <span class="small-muted" style="margin-left:35px">{desc}</span>
                </div>""",
                unsafe_allow_html=True
            )

    with right:
        section_heading("02 / WORKSPACE", "Start analysing")
        if rgb_image is not None:
            st.image(rgb_image, caption="Current uploaded image", use_container_width=True)
            st.success("Image loaded. Select Image Detection or Segmentation from the sidebar.")
        else:
            st.markdown("""
            <div class="colour-tile" style="padding:25px;text-align:center;">
              <div style="font-size:2.7rem;">🖼️</div>
              <h3 style="color:#eaf4ff">Your image workspace is ready</h3>
              <p class="small-muted">Upload an image using the sidebar to begin.</p>
            </div>
            """, unsafe_allow_html=True)
        st.info("Tip: use a clear image with distinct coloured objects and a contrasting background.")

    st.markdown("---")
    section_heading("03 / DETECTION RANGE", "Ten supported colour classes")
    names = list(COLOR_RANGES.keys())
    colour_cols = st.columns(5)
    for idx, colour in enumerate(names):
        with colour_cols[idx % 5]:
            st.markdown(
                f"""<div class="colour-tile">
                <span class="colour-dot" style="background:{COLOR_HEX.get(colour, '#fff')}"></span>
                <b>{COLOR_EMOJI.get(colour, '')} {colour}</b>
                </div>""",
                unsafe_allow_html=True
            )

    with st.expander("View technical pipeline"):
        st.code(
            "Input Image\n   ↓\nImage Preprocessing\n   ↓\nBGR to HSV Conversion\n   ↓\nHSV Colour Thresholding\n   ↓\nBinary Masks\n   ↓\nMorphological Operations\n   ↓\nContour Detection\n   ↓\nObject Geometry + Analytics",
            language="text"
        )


# =========================================================
# IMAGE DETECTION
# =========================================================
elif page == "Image Detection":
    section_heading("01 / DETECT", "Multi-colour object detection",
                    "Locate colour regions and inspect their measured properties.")
    if bgr_image is None:
        st.warning("Upload an image from the sidebar to start detection.")
    else:
        with st.spinner("Analysing image and detecting colour regions..."):
            start = time.perf_counter()
            output, masks, objects = get_detection(bgr_image, min_area)
            elapsed = (time.perf_counter() - start) * 1000

        show_image_pair(rgb_image, output, "Original image", "Detection overlay", right_is_bgr=True)

        st.markdown("")
        a, b, c, d = st.columns(4)
        a.metric("Objects detected", len(objects))
        b.metric("Processing time", f"{elapsed:.2f} ms")
        d.metric("Image size", f"{bgr_image.shape[1]} × {bgr_image.shape[0]}")
        c.metric("Processing throughput", f"{1000 / elapsed:.1f} FPS" if elapsed > 0 else "—")
        st.caption("Throughput is measured for this uploaded-image processing run; it is not webcam capture FPS.")

        if objects:
            df = pd.DataFrame(objects)
            section_heading("02 / OBJECT DATA", "Detection results")
            st.dataframe(df, use_container_width=True, hide_index=True)

            counts = df["Color"].value_counts().reindex(list(COLOR_RANGES.keys()), fill_value=0)
            fig = px.bar(
                x=counts.index, y=counts.values, color=counts.index,
                color_discrete_map=COLOR_HEX, labels={"x": "Colour", "y": "Objects"},
                title="Detected objects by colour"
            )
            st.plotly_chart(plot_layout(fig, 320), use_container_width=True)

            csv_data = export_detection_report(objects)
            if csv_data:
                st.download_button(
                    "⬇️ Download detection CSV",
                    data=csv_data,
                    file_name="chromovision_detection_report.csv",
                    mime="text/csv"
                )

            if st.button("＋ Save this run to Detection History", type="primary"):
                st.session_state.detection_history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Objects": len(objects),
                    "Processing Time (ms)": round(elapsed, 2),
                    "Image": uploaded_file.name if uploaded_file else "Uploaded image"
                })
                st.success("This detection run has been added to the current session history.")
        else:
            st.warning("No objects met the current thresholds. Try a different image or adjust Minimum object area.")


# =========================================================
# SEGMENTATION
# =========================================================
elif page == "Segmentation":
    section_heading("02 / SEGMENT", "Colour segmentation laboratory",
                    "Inspect the binary mask and extracted pixels for each supported colour.")
    if bgr_image is None:
        st.warning("Upload an image from the sidebar first.")
    else:
        _, masks, objects = get_detection(bgr_image, min_area)
        selected = st.selectbox("Choose colour mask", list(masks.keys()))
        mask = masks[selected]
        extracted = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
        show_image_pair(mask, extracted, f"{selected} binary mask", f"{selected} extracted region", right_is_bgr=True)

        foreground = cv2.countNonZero(mask)
        total_pixels = mask.size
        percentage = foreground / total_pixels * 100 if total_pixels else 0
        x1, x2, x3 = st.columns(3)
        x1.metric("Foreground pixels", f"{foreground:,}")
        x2.metric("Image pixels", f"{total_pixels:,}")
        x3.metric("Foreground ratio", f"{percentage:.2f}%")
        st.caption("Foreground ratio is the share of pixels included in this colour mask; it is not object-detection accuracy.")

        with st.expander("Inspect all colour masks"):
            for colour, colour_mask in masks.items():
                with st.container():
                    st.markdown(f"**{COLOR_EMOJI.get(colour, '')} {colour}**")
                    st.image(colour_mask, use_container_width=True)
                    st.divider()


# =========================================================
# HSV CALIBRATION
# =========================================================
elif page == "HSV Calibration":
    section_heading("03 / CALIBRATE", "Interactive HSV calibration",
                    "Adjust the selected colour threshold and immediately inspect its segmentation result.")
    if bgr_image is None:
        st.warning("Upload an image from the sidebar first.")
    else:
        selected_color = st.selectbox("Colour to calibrate", list(COLOR_RANGES.keys()))
        defaults = COLOR_RANGES[selected_color]
        default_lower, default_upper = defaults[0]
        st.markdown(f"### {COLOR_EMOJI.get(selected_color, '')} {selected_color} threshold controls")

        if selected_color == "White":
            sat_max = st.slider("Maximum saturation", 0, 255, int(default_upper[1]), key="white_sat")
            val_min = st.slider("Minimum brightness", 0, 255, int(default_lower[2]), key="white_val")
            lower = np.array([0, 0, val_min], dtype=np.uint8)
            upper = np.array([179, sat_max, 255], dtype=np.uint8)
        elif selected_color == "Black":
            val_max = st.slider("Maximum brightness", 0, 255, int(default_upper[2]), key="black_val")
            lower = np.array([0, 0, 0], dtype=np.uint8)
            upper = np.array([179, 255, val_max], dtype=np.uint8)
        else:
            h_min, h_max = st.slider(
                "Hue range", 0, 179,
                (int(default_lower[0]), int(default_upper[0])),
                key=f"hue_{selected_color}"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                s_min = st.slider("Minimum saturation", 0, 255, int(default_lower[1]), key=f"sat_{selected_color}")
            with col_b:
                v_min = st.slider("Minimum brightness", 0, 255, int(default_lower[2]), key=f"val_{selected_color}")
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, 255, 255], dtype=np.uint8)

        custom_ranges = {selected_color: [(lower, upper)]}
        output, custom_masks, custom_objects = detect_colors(
            bgr_image, min_area=min_area, custom_ranges=custom_ranges
        )
        calibrated_mask = custom_masks[selected_color]
        extracted = cv2.bitwise_and(bgr_image, bgr_image, mask=calibrated_mask)

        show_image_pair(calibrated_mask, extracted, "Custom binary mask", "Extracted colour pixels", right_is_bgr=True)
        st.markdown("**Detection preview**")
        st.image(get_rgb_from_bgr(output), use_container_width=True)

        q1, q2, q3 = st.columns(3)
        q1.metric("Objects detected", len(custom_objects))
        q2.metric("Mask pixels", f"{cv2.countNonZero(calibrated_mask):,}")
        q3.metric("Selected colour", selected_color)
        if selected_color == "Red":
            st.info("This calibration preview uses the first red hue band. The main detector may use two hue bands to cover red near both ends of the HSV hue scale.")
        st.caption("Calibration changes apply to this preview only; they do not permanently overwrite the module's default colour ranges.")


# =========================================================
# NOISE ANALYSIS
# =========================================================
elif page == "Noise Analysis":
    section_heading("04 / FILTER", "Noise analysis & denoising",
                    "Compare the image before and after a selected smoothing filter.")
    if bgr_image is None:
        st.warning("Upload an image from the sidebar first.")
    else:
        method = st.selectbox("Denoising method", ["Gaussian", "Median", "Bilateral"])
        colour = st.selectbox("Colour to inspect", list(COLOR_RANGES.keys()))
        processed, filter_ms = apply_denoising(bgr_image, method)
        _, masks_before, _ = get_detection(bgr_image, min_area)
        _, masks_after, _ = get_detection(processed, min_area)
        before = calculate_noise_ratio(masks_before[colour])
        after = calculate_noise_ratio(masks_after[colour])
        show_image_pair(bgr_image, processed, "Original image", f"{method} filtered image", left_is_bgr=True, right_is_bgr=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Noise proxy before", f"{before:.2f}%")
        c2.metric("Noise proxy after", f"{after:.2f}%")
        c3.metric("Filter processing time", f"{filter_ms:.2f} ms")
        change = before - after
        if change > 0:
            st.success(f"The small-component noise proxy decreased by {change:.2f} percentage points.")
        elif change < 0:
            st.info(f"The small-component noise proxy increased by {abs(change):.2f} percentage points for this image.")
        else:
            st.info("The noise proxy did not change for this image.")
        st.caption("This is a small-connected-component proxy calculated from the colour mask, not a ground-truth measurement of image noise or denoising accuracy.")


# =========================================================
# LIGHTING EXPERIMENTS
# =========================================================
elif page == "Lighting Experiments":
    section_heading("05 / EXPERIMENT", "Brightness robustness experiment",
                    "Observe how simulated brightness changes affect colour segmentation.")
    if bgr_image is None:
        st.warning("Upload an image from the sidebar first.")
    else:
        brightness = st.slider("Brightness adjustment", -100, 100, 0, step=5)
        adjusted = cv2.convertScaleAbs(bgr_image, alpha=1.0, beta=brightness)
        output, masks, objects = get_detection(adjusted, min_area)
        show_image_pair(adjusted, output, "Brightness-adjusted input", "Detection result", left_is_bgr=True, right_is_bgr=True)
        a, b, c = st.columns(3)
        a.metric("Brightness offset", f"{brightness:+d}")
        b.metric("Detected objects", len(objects))
        c.metric("Colour classes present", len(set(item["Color"] for item in objects)) if objects else 0)
        st.caption("This experiment simulates brightness changes digitally. For a complete lighting study, also test images captured under real lighting conditions.")


# =========================================================
# PERFORMANCE
# =========================================================
elif page == "Performance":
    section_heading("06 / BENCHMARK", "Processing performance",
                    "Run repeated image-processing trials and inspect timing statistics.")
    if bgr_image is None:
        st.warning("Upload an image from the sidebar first.")
    else:
        iterations = st.slider("Benchmark iterations", 5, 50, 10, step=5)
        if st.button("▶ Run benchmark", type="primary"):
            with st.spinner(f"Running {iterations} processing iterations..."):
                results = benchmark_processing(
                    bgr_image,
                    lambda img: detect_colors(img, min_area=min_area),
                    iterations=iterations
                )
            st.session_state["benchmark_results"] = results

        results = st.session_state.get("benchmark_results")
        if results:
            a, b, c = st.columns(3)
            a.metric("Average time", f"{results['Average Processing Time (ms)']} ms")
            b.metric("Processing throughput", f"{results['Processing FPS']} FPS")
            c.metric("Maximum time", f"{results['Maximum Processing Time (ms)']} ms")
            st.json(results)
            st.info("Processing FPS is derived from image-processing time. It is not a live webcam frame rate.")
        else:
            st.info("Set the number of iterations and run the benchmark to view results.")


# =========================================================
# ANALYTICS
# =========================================================
elif page == "Analytics":
    section_heading("07 / INSIGHTS", "Detection analytics dashboard",
                    "Explore colour distribution, object geometry and downloadable results.")
    if bgr_image is None:
        st.warning("Upload an image from the sidebar first.")
    else:
        _, masks, objects = get_detection(bgr_image, min_area)
        if not objects:
            st.warning("No objects detected. Adjust the minimum area or upload another image.")
        else:
            df = pd.DataFrame(objects)
            summary = analyze_detection_data(objects)
            a, b, c = st.columns(3)
            a.metric("Total objects", summary["Total Objects"])
            b.metric("Average area", f"{summary['Average Area']}")
            c.metric("Largest object", f"{summary['Largest Object']}")

            counts = df["Color"].value_counts().reindex(list(COLOR_RANGES.keys()), fill_value=0)
            chart_left, chart_right = st.columns(2, gap="large")
            with chart_left:
                fig = px.pie(
                    names=counts[counts > 0].index,
                    values=counts[counts > 0].values,
                    color=counts[counts > 0].index,
                    color_discrete_map=COLOR_HEX,
                    hole=.55,
                    title="Colour distribution"
                )
                st.plotly_chart(plot_layout(fig, 370), use_container_width=True)
            with chart_right:
                fig2 = px.bar(
                    x=counts.index, y=counts.values, color=counts.index,
                    color_discrete_map=COLOR_HEX,
                    labels={"x": "Colour", "y": "Object count"},
                    title="Object count by colour"
                )
                st.plotly_chart(plot_layout(fig2, 370), use_container_width=True)

            section_heading("GEOMETRY", "Object measurements")
            geometry = []
            for colour, mask in masks.items():
                for item in analyze_contours(mask, min_area=min_area):
                    item["Color"] = colour
                    geometry.append(item)
            if geometry:
                geometry_df = pd.DataFrame(geometry)
                st.dataframe(geometry_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No contour geometry was returned for the current thresholds.")

            area_fig = px.bar(
                df, x="Color", y="Area", color="Color",
                color_discrete_map=COLOR_HEX,
                title="Detected object area comparison",
                hover_data=[col for col in ["ID", "X", "Y", "Width", "Height"] if col in df.columns]
            )
            st.plotly_chart(plot_layout(area_fig, 350), use_container_width=True)

            csv_data = export_detection_report(objects)
            if csv_data:
                st.download_button(
                    "⬇️ Export full detection report (CSV)",
                    data=csv_data,
                    file_name="chromovision_analytics_report.csv",
                    mime="text/csv"
                )
            st.caption("Charts describe the detections produced by the current HSV thresholds; they do not represent validated model accuracy.")


# =========================================================
# DETECTION HISTORY
# =========================================================
elif page == "Detection History":
    section_heading("08 / ACTIVITY", "Detection session history",
                    "Saved runs from the current Streamlit session.")
    history = st.session_state.detection_history
    if history:
        history_df = pd.DataFrame(history)
        a, b = st.columns(2)
        a.metric("Saved sessions", len(history))
        b.metric("Objects across saved runs", int(history_df["Objects"].sum()))
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download history (CSV)",
            data=history_df.to_csv(index=False),
            file_name="chromovision_detection_history.csv",
            mime="text/csv"
        )
        if st.button("Clear session history"):
            st.session_state.detection_history = []
            st.success("Session history cleared.")
            st.rerun()
        st.caption("History is stored in Streamlit session memory and will not persist after the session ends or the app restarts.")
    else:
        st.info("No saved runs yet. Open Image Detection and save a detection result to populate this page.")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
footer_left, footer_right = st.columns([3, 1])
with footer_left:
    st.caption("🎨 CHROMOVISION AI · Intelligent Multi-Colour Object Detection & Image Analysis")
    st.caption("Built with Python · OpenCV · NumPy · Pandas · Streamlit · Plotly")
with footer_right:
    st.markdown(
        '<div style="text-align:right;color:#6b819f;font-size:.78rem;">ACADEMIC PROJECT PROTOTYPE</div>',
        unsafe_allow_html=True
    )