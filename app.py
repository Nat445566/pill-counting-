import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd

# --- [UNCHANGED] Core Helper Function: Get Pill Properties ---
def get_pill_properties(image_bgr, contour):
    """A definitive, hierarchical classifier for shape and color."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = "Unknown"
    if perimeter > 0:
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if circularity > 0.82 and aspect_ratio < 1.4: shape = "Round"
        elif aspect_ratio > 2.0: shape = "Capsule"
        elif len(approx) == 4: shape = "Rectangular"
        else: shape = "Oval"

    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    eroded_mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=eroded_mask)[:3]
    h, s, v = mean_hsv
    if s < 45:
        if v > 150: color = "White"
        elif v < 70: color = "Black"
        else: color = "Gray"
    else:
        if (h <= 10 or h >= 165): color = "Red" if s > 120 else "Pink"
        elif h <= 25: color = "Brown" if v < 180 else "Orange"
        elif h <= 40: color = "Yellow"
        elif h <= 85: color = "Green"
        elif h <= 130: color = "Blue"
        else: color = "Unknown"
    return shape, color

# --- [KEY FUNCTION] Central Pill Filtering and Classification ---
def filter_and_classify_pills(image, contours, params):
    """Applies your original, robust filtering logic to any list of contours."""
    detected_pills = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (params['min_area'] < area < params['max_area']):
            continue
        hull = cv2.convexHull(c)
        if hull.shape[0] < 3: continue
        solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        if solidity < 0.9:
            continue
        shape, color = get_pill_properties(image, c)
        if color == "Unknown" or shape == "Unknown":
            continue
        detected_pills.append({'shape': shape, 'color': color, 'contour': c})
    return detected_pills

# --- Detector Functions (Now all are "General Detectors") ---
def get_contours_adaptive_color(image, params):
    """Your original, proven adaptive contour detection algorithm."""
    def is_background_light(img):
        h, w, _ = img.shape
        corner_size = int(min(h, w) * 0.1)
        corners = [img[0:corner_size, 0:corner_size], img[0:corner_size, w-corner_size:w], img[h-corner_size:h, 0:corner_size], img[h-corner_size:h, w-corner_size:w]]
        return np.mean([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean() for c in corners]) > 120
    def detect_on_dark_bg(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        blurred = cv2.GaussianBlur(l, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3), cv2.MORPH_OPEN, kernel, iterations=2)
    def detect_on_light_bg(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 40, 50]), np.array([180, 255, 255]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        final_mask = cv2.bitwise_or(mask, cv2.dilate(white_mask, np.ones((3,3), np.uint8), iterations=2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3), cv2.MORPH_OPEN, kernel, iterations=2)
    final_mask = detect_on_light_bg(image) if is_background_light(image) else detect_on_dark_bg(image)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_contours_canny(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, params['canny_thresh1'], params['canny_thresh2'])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_contours_watershed(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    markers = cv2.watershed(image, cv2.connectedComponents(np.uint8(sure_fg))[1] + 1)
    all_contours = []
    for label in np.unique(markers):
        if label <= 1: continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
    return all_contours

def get_contours_hough_circles(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=params['hough_min_dist'],
                              param1=params['hough_param1'], param2=params['hough_param2'],
                              minRadius=params['hough_min_radius'], maxRadius=params['hough_max_radius'])
    contours = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center, radius = (i[0], i[1]), i[2]
            t = np.arange(0, 2 * np.pi, 0.1)
            pts = np.array([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]).T.astype(np.int32)
            contours.append(pts.reshape((-1, 1, 2)))
    return contours

def get_contours_blob_detector(image, params):
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.minArea, blob_params.maxArea = params['min_area'], params['max_area']
    blob_params.filterByCircularity = True
    blob_params.minCircularity = params['blob_min_circularity']
    blob_params.filterByConvexity = True
    blob_params.minConvexity = 0.87
    blob_params.filterByInertia = True
    blob_params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(image)
    contours = []
    for kp in keypoints:
        center, radius = (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2)
        t = np.arange(0, 2 * np.pi, 0.1)
        pts = np.array([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]).T.astype(np.int32)
        contours.append(pts.reshape((-1, 1, 2)))
    return contours

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide")
st.title("Pharmaceutical Tablet Analysis System")

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Controls")
    detector_options = {
        "Contour-Based (Original Accurate)": get_contours_adaptive_color,
        "Edge-Based (Canny)": get_contours_canny,
        "Watershed Segmentation": get_contours_watershed,
        "Hough Circle Transform": get_contours_hough_circles,
        "Simple Blob Detector": get_contours_blob_detector
    }
    detector_name = st.selectbox("1. Select Detector Algorithm", detector_options.keys())
    analysis_mode = st.radio("2. Select Analysis Mode", ("Full Image Detection", "Manual ROI (Matching Pills)"))
    
    with st.expander("🔬 Tuning & Advanced Options"):
        min_area = st.slider("Min Pill Area", 50, 5000, 500, key="min_area_slider")
        max_area = st.slider("Max Pill Area", 5000, 100000, 50000, key="max_area_slider")
        params = {'min_area': min_area, 'max_area': max_area}
        
        if detector_name == "Edge-Based (Canny)":
            params['canny_thresh1'] = st.slider("Canny Threshold 1", 0, 255, 30)
            params['canny_thresh2'] = st.slider("Canny Threshold 2", 0, 255, 150)
        elif detector_name == "Hough Circle Transform":
            st.markdown("##### Hough Circle Tuning")
            params['hough_min_dist'] = st.slider("Min Distance", 10, 100, 20)
            params['hough_param1'] = st.slider("Canny Edge Upper Threshold", 50, 250, 100)
            params['hough_param2'] = st.slider("Accumulator Threshold", 10, 100, 30)
            params['hough_min_radius'] = st.slider("Min Radius", 5, 50, 10)
            params['hough_max_radius'] = st.slider("Max Radius", 50, 200, 80)
        elif detector_name == "Simple Blob Detector":
            st.markdown("##### Blob Detector Tuning")
            params['blob_min_circularity'] = st.slider("Min Circularity", 0.1, 1.0, 0.8, 0.05)

# --- Main Page Layout ---
_, main_col, _ = st.columns([1, 2, 1])
with main_col:
    st.info("Upload an image, then select your algorithm and mode from the sidebar.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert('RGB')
        orig_img = np.array(pil_img)
        scale = 800 / orig_img.shape[1]
        new_size = (int(orig_img.shape[1] * scale), int(orig_img.shape[0] * scale))
        st.session_state.img = cv2.cvtColor(cv2.resize(orig_img, new_size), cv2.COLOR_RGB2BGR)

    if 'img' in st.session_state and st.session_state.img is not None:
        st.subheader("Image Analysis")
        
        if analysis_mode == "Manual ROI (Matching Pills)":
            st.warning("Draw a box on the image to define the target pill for matching.")
            cropped_pil = st_cropper(Image.fromarray(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB)), realtime_update=True, box_color='lime')
        else:
            st.image(st.session_state.img, channels="BGR", caption="Full image ready for analysis.")

        st.divider()

        button_label = "Run Full Image Detection" if analysis_mode == "Full Image Detection" else "Find All Matching Pills"
        if st.button(button_label, use_container_width=True):
            with st.spinner("Analyzing..."):
                if analysis_mode == "Full Image Detection":
                    contours = detector_options[detector_name](st.session_state.img, params)
                    detected_pills = filter_and_classify_pills(st.session_state.img, contours, params)
                    annotated_image = st.session_state.img.copy()
                    for pill in detected_pills:
                        x,y,w,h = cv2.boundingRect(pill['contour'])
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(annotated_image, f"{pill['shape']}, {pill['color']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    st.subheader("Detection Results")
                    st.metric("Total Pills Found", len(detected_pills))
                    st.image(annotated_image, channels="BGR")
                    if detected_pills:
                        df = pd.DataFrame(detected_pills).drop(columns='contour')
                        summary_df = df.groupby(['shape', 'color']).size().reset_index(name='quantity')
                        st.dataframe(summary_df, use_container_width=True)

                elif analysis_mode == "Manual ROI (Matching Pills)":
                    roi = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                    if roi.size < 100:
                        st.error("Please draw a valid box on the image.")
                    else:
                        roi_params = {'min_area': 10, 'max_area': roi.shape[0] * roi.shape[1], **params}
                        roi_contours = detector_options[detector_name](roi, roi_params)
                        target_pills = filter_and_classify_pills(roi, roi_contours, roi_params)
                        if not target_pills:
                            st.error("Could not identify a valid pill in the selected ROI.")
                        else:
                            target = target_pills[0]
                            all_contours = detector_options[detector_name](st.session_state.img, params)
                            all_pills = filter_and_classify_pills(st.session_state.img, all_contours, params)
                            matches = [p for p in all_pills if p['shape'] == target['shape'] and p['color'] == target['color']]
                            annotated_image = st.session_state.img.copy()
                            for pill in matches:
                                x,y,w,h = cv2.boundingRect(pill['contour'])
                                cv2.rectangle(annotated_image, (x,y), (x+w,y+h), (0,255,255), 3)
                            
                            st.subheader("Matching Results")
                            st.metric(f"Found {len(matches)} matches for:", f"{target['shape']}, {target['color']}")
                            st.image(annotated_image, channels="BGR")
                            
                            # --- [NEW] ADDED SUMMARY DATAFRAME FOR ROI MATCHING ---
                            match_data = {
                                'Target Shape': [target['shape']],
                                'Target Color': [target['color']],
                                'Quantity Found': [len(matches)]
                            }
                            st.dataframe(pd.DataFrame(match_data), use_container_width=True)

    elif not uploaded_file:
         st.info("Awaiting image upload to begin.")
