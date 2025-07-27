import cv2
import numpy as np
import streamlit as st
from collections import Counter
import os

class BaccaratRoadmapAnalyzer:
    def __init__(self):
        self.history = []

    def add_result(self, result):
        result = result.upper()
        if result in ['B', 'P', 'T']:
            self.history.append(result)

    def get_big_road(self):
        road = []
        col = []
        last = ''
        for result in self.history:
            if result == 'T':
                continue
            if result == last:
                col.append(result)
            else:
                if col:
                    road.append(col)
                col = [result]
                last = result
        if col:
            road.append(col)
        return road

    def analyze_streaks(self):
        road = self.get_big_road()
        if not road:
            return "Not enough data", 50
        last_col = road[-1]
        current_result = last_col[-1]
        streak_length = len(last_col)
        if streak_length >= 3:
            return current_result, min(100, 70 + streak_length * 5)
        else:
            return self.bias_prediction()

    def bias_prediction(self):
        counter = Counter([r for r in self.history if r in ['B', 'P']])
        b_count = counter['B']
        p_count = counter['P']
        total = max(1, b_count + p_count)
        if b_count > p_count:
            return 'B', 50 + int((b_count - p_count) / total * 50)
        else:
            return 'P', 50 + int((p_count - b_count) / total * 50)

    def derived_road_prediction(self):
        road = self.get_big_road()
        if len(road) < 4:
            return self.analyze_streaks()
        len_diffs = [len(road[i]) - len(road[i-1]) for i in range(1, len(road))]
        momentum = sum(len_diffs[-3:])
        if momentum > 0:
            return road[-1][-1], 75
        else:
            alt = 'B' if road[-1][-1] == 'P' else 'P'
            return alt, 75

    def predict_next(self):
        if len(self.history) < 6:
            return self.analyze_streaks()
        return self.derived_road_prediction()

    def save_history(self, path="history.txt"):
        with open(path, "w") as f:
            f.write(" ".join(self.history))

    def load_history(self, path="history.txt"):
        if os.path.exists(path):
            with open(path, "r") as f:
                data = f.read().strip().split()
                self.history = data

def extract_big_road_from_image(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([100, 150, 50])
    blue_upper = np.array([130, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    def find_dots(mask, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 5:
                results.append((label, (x, y)))
        return results

    red_dots = find_dots(red_mask, 'P')
    blue_dots = find_dots(blue_mask, 'B')
    all_dots = red_dots + blue_dots
    all_dots.sort(key=lambda d: (d[1][0] // 20, d[1][1]))
    return [d[0] for d in all_dots]

def run_prediction_from_screenshot(image_path):
    analyzer = BaccaratRoadmapAnalyzer()
    analyzer.load_history()
    extracted = extract_big_road_from_image(image_path)
    for outcome in extracted:
        analyzer.add_result(outcome)
    prediction, confidence = analyzer.predict_next()
    analyzer.save_history()
    return analyzer.history, prediction, confidence

# Streamlit GUI
st.title("Baccarat Screenshot Analyzer")
st.write("Upload a screenshot of the Big Road to get the next best move.")

uploaded_file = st.file_uploader("Choose a screenshot image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    with open("temp_upload.png", "wb") as f:
        f.write(uploaded_file.read())
    history, prediction, confidence = run_prediction_from_screenshot("temp_upload.png")
    st.subheader("Extracted History")
    st.write(" ".join(history))
    st.success(f"Prediction: {prediction} (Confidence: {confidence}%)")
    with open("history.txt", "r") as h:
        st.download_button("Download Game History", h.read(), file_name="history.txt")
