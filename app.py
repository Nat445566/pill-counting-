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

# --- Detector Functions (Part 1: Contour Generators) ---
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

# --- Detector Functions (Part 2: Template Matchers) ---
def find_template_matches(image, template, params):
    if template.size < 100: return []
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= params['match_threshold'])
    rects = [[int(pt[0]), int(pt[1]), int(w), int(h)] for pt in zip(*loc[::-1])]
    rects, _ = cv2.groupRectangles(rects * 2, 1, 0.2)
    return rects

def find_feature_match(image, template):
    if template.size < 100: return None
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(main_gray, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2: return None
    flann = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(--1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None: return None
        h, w = template_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        return np.int32(cv2.perspectiveTransform(pts, M))
    return None

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
        "Template Matching": find_template_matches,
        "Feature-Based Matching": find_feature_match
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
        if detector_name == "Template Matching":
            params['match_threshold'] = st.slider("Match Confidence", 0.5, 1.0, 0.8)

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
        
        is_contour_detector = detector_name in ["Contour-Based (Original Accurate)", "Edge-Based (Canny)", "Watershed Segmentation"]
        is_template_detector = detector_name in ["Template Matching", "Feature-Based Matching"]
        
        # --- UI Setup: Decide whether to show cropper based on mode/detector ---
        # The cropper is ALWAYS shown for Manual ROI mode.
        # For Full Image mode, it's ONLY shown if a template detector is selected.
        show_cropper = (analysis_mode == "Manual ROI (Matching Pills)") or \
                       (analysis_mode == "Full Image Detection" and is_template_detector)

        if show_cropper:
            if analysis_mode == "Full Image Detection":
                 st.info(f"'{detector_name}' requires a template. Please draw a box around a single pill to search for in the full image.")
            else: # Manual ROI mode
                 st.warning("Draw a box on the image to define the target pill for matching.")
            cropped_pil = st_cropper(Image.fromarray(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB)), realtime_update=True, box_color='lime')
        else: # Full Image mode with a contour detector
            st.image(st.session_state.img, channels="BGR", caption="Full image ready for analysis.")

        st.divider()

        # --- Execution Logic ---
        button_label = "Run Analysis"
        if analysis_mode == "Full Image Detection":
            button_label = "Run Full Image Detection"
            if is_template_detector:
                button_label = "Find All Matches in Full Image"
        elif analysis_mode == "Manual ROI (Matching Pills)":
            button_label = "Find All Matching Pills"

        if st.button(button_label, use_container_width=True):
            with st.spinner("Analyzing..."):
                # --- A: Full Image Detection Workflow ---
                if analysis_mode == "Full Image Detection":
                    if is_contour_detector:
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
                            st.dataframe(df.groupby(['shape', 'color']).size().reset_index(name='quantity'))
                    
                    elif is_template_detector: # Guided template matching
                        template = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                        if template.size < 100:
                            st.error("Please draw a valid box on the image to use as a template.")
                        else:
                            annotated_image = st.session_state.img.copy()
                            if detector_name == "Template Matching":
                                matches = find_template_matches(st.session_state.img, template, params)
                                for (x, y, w, h) in matches:
                                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255,0,0), 3)
                                st.metric("Template Matches Found", len(matches))
                            elif detector_name == "Feature-Based Matching":
                                match_poly = find_feature_match(st.session_state.img, template)
                                st.metric("Feature Matches Found", 1 if match_poly is not None else 0)
                                if match_poly is not None:
                                    cv2.polylines(annotated_image, [match_poly], True, (255,0,255), 3)
                            st.subheader("Matching Results")
                            st.image(annotated_image, channels="BGR")

                # --- B: Manual ROI (Matching Pills) Workflow ---
                elif analysis_mode == "Manual ROI (Matching Pills)":
                    roi = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                    if roi.size < 100:
                        st.error("Please draw a valid box on the image.")
                    else:
                        if is_contour_detector:
                            roi_params = {'min_area': 10, 'max_area': roi.shape[0] * roi.shape[1]}
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

                        elif is_template_detector: # This logic is identical to the guided full-image search
                            annotated_image = st.session_state.img.copy()
                            if detector_name == "Template Matching":
                                matches = find_template_matches(st.session_state.img, roi, params)
                                for (x, y, w, h) in matches:
                                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255,0,0), 3)
                                st.metric("Template Matches Found", len(matches))
                            elif detector_name == "Feature-Based Matching":
                                match_poly = find_feature_match(st.session_state.img, roi)
                                st.metric("Feature Matches Found", 1 if match_poly is not None else 0)
                                if match_poly is not None:
                                    cv2.polylines(annotated_image, [match_poly], True, (255,0,255), 3)
                            st.subheader("Matching Results")
                            st.image(annotated_image, channels="BGR")

    elif not uploaded_file:
         st.info("Awaiting image upload to begin.")
