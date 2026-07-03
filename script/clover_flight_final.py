import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from clover import srv
from std_srvs.srv import Trigger
from clover.srv import SetLEDEffect
import math
import os

rospy.init_node('flight')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)

bridge = CvBridge()
latest_frame = None

last_saved_x = None
last_saved_y = None

def image_callback(msg):
    global latest_frame
    latest_frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

rospy.Subscriber('main_camera/image_raw', Image, image_callback, queue_size=1)
BLUE_LOW = (90, 50, 40)
BLUE_HIGH = (140, 255, 255)
MIN_AREA = 500

GREEN_LOW = (35, 50, 50)
GREEN_HIGH = (85, 255, 255)
MIN_TRASH_AREA = 20
MAX_TRASH_AREA = 3000   

NUM_PANEL = 1

set_led_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)
LED_COLORS = {
    'white':  (255, 255, 255),
    'orange': (255, 140, 0),
    'yellow': (255, 255, 0),
    'red':    (255, 0, 0),
}

COLOR_STATUS = {
    'orange': 'некритический перегрев',
    'yellow': 'нормальное состояние',
    'red': 'срочный ремонт'
}

REPORT_DIR = '/home/clover/Desktop/solar/Logs'
IMAGES_DIR = '/home/clover/Desktop/solar/Panels'
ANNOTATED_DIR = '/home/clover/Desktop/solar/Annotated'

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

report = open(os.path.join(REPORT_DIR, 'panels_report.txt'), 'a', encoding='utf-8')

def set_led(color_name):
    r, g, b = LED_COLORS[color_name]
    try:
        set_led_effect(r=r, g=g, b=b)
    except rospy.ServiceException as e:
        print("LED service call failed:", e)

COLOR_RANGES = {
    'orange': [((10, 100, 100), (20, 255, 255))],
    'yellow': [((21, 100, 100), (35, 255, 255))],
    'red':    [((0, 100, 100), (9, 255, 255)), ((170, 100, 100), (180, 255, 255))],
}
MIN_INDICATOR_AREA = 200

def navigate_wait(x=0, y=0, z=1, speed=1, frame_id='aruco_map', auto_arm=False, tolerance=0.2):
    navigate(x=x, y=y, z=z, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')

        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break

        rospy.sleep(0.2)

def get_panel_bbox():
    if latest_frame is None:
        return None
    hsv = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(BLUE_LOW), np.array(BLUE_HIGH))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None
    return cv2.boundingRect(largest)


def get_panel_crop(bbox, padding=110):
    x, y, w, h = bbox
    img_h, img_w = latest_frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return latest_frame[y1:y2, x1:x2]

def detect_indicator_color(bbox, search_margin=110):
    x, y, w, h = bbox
    img_h, img_w = latest_frame.shape[:2]
    x1 = max(0, x - search_margin)
    y1 = max(0, y - search_margin)
    x2 = min(img_w, x + w + search_margin)
    y2 = min(img_h, y + h + search_margin)

    roi = latest_frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    best_color, best_area = None, 0
    for color_name, ranges in COLOR_RANGES.items():
        mask = None
        for low, high in ranges:
            m = cv2.inRange(hsv, np.array(low), np.array(high))
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        area = cv2.contourArea(max(contours, key=cv2.contourArea))
        if area > best_area:
            best_area = area
            best_color = color_name

    if best_area < MIN_INDICATOR_AREA:
        return None
    return best_color

def panel_visible():
    if latest_frame is None:
        return False
    hsv = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(BLUE_LOW), np.array(BLUE_HIGH))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    return cv2.contourArea(max(contours, key=cv2.contourArea)) > MIN_AREA

def is_new_panel(x, y, threshold=1.3):
    global last_saved_x, last_saved_y
    if last_saved_x is None or last_saved_y is None:
        return True
    return math.hypot(x - last_saved_x, y - last_saved_y) > threshold

def find_trash_objects(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(GREEN_LOW), np.array(GREEN_HIGH))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_TRASH_AREA <= area <= MAX_TRASH_AREA:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
    return boxes

def inspect_if_panel():
    global last_saved_x, last_saved_y, NUM_PANEL

    bbox = get_panel_bbox()
    if bbox is None:
        return
    
    telem = get_telemetry(frame_id='aruco_map')
    x, y, z = telem.x, telem.y, telem.z

    if is_new_panel(x, y):
        navigate_wait(x=x, y=y, z=z+0.6, speed=1, frame_id='aruco_map')
        rospy.sleep(0.5)
        color = detect_indicator_color(bbox)
        status = COLOR_STATUS.get(color, 'unknown')
        if color is not None:
            set_led(color)

        crop = get_panel_crop(bbox)
        filename = "panel_x{:.2f}_y{:.2f}.jpg".format(x, y)
        filepath = os.path.join(IMAGES_DIR, filename)
        cv2.imwrite(filepath, crop)

        boxes = find_trash_objects(crop)
        contamination = len(boxes)

        annotated = crop.copy()
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        annotated_path = os.path.join(ANNOTATED_DIR, filename)
        cv2.imwrite(annotated_path, annotated)

        report.write(f"Солнечная панель № {NUM_PANEL}: "+"{:.2f} {:.2f}".format(x, y)+", "+str(status)+", "+str(contamination)+"\n")
        print(f"Solar panel number {NUM_PANEL}: "+"{:.2f} {:.2f}".format(x, y)+", "+str(color)+", "+str(contamination)+"\n")
        NUM_PANEL+=1
        report.flush()

        last_saved_x, last_saved_y = x, y
        rospy.sleep(5)
        set_led('white')
        navigate_wait(x=x, y=y, z=z, speed=1, frame_id='aruco_map')

def main():
    set_led('white')
    navigate_wait(z=1, speed=1, frame_id='body', auto_arm=True)

    waypoints = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
        (9, 1), (8, 1), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2),
        (9, 3), (8, 3), (7, 3), (6, 3), (5, 3), (4, 3), (3, 3), (2, 3), (1, 3), (0, 3),
        (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4),
        (9, 5), (8, 5), (7, 5), (6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5),
        (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
        (9, 7), (8, 7), (7, 7), (6, 7), (5, 7), (4, 7), (3, 7), (2, 7), (1, 7), (0, 7),
        (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8),
        (9, 9), (8, 9), (7, 9), (6, 9), (5, 9), (4, 9), (3, 9), (2, 9), (1, 9), (0, 9),
    ]
    for wx, wy in waypoints:
        navigate_wait(x=wx, y=wy, z=1, speed=1, frame_id='aruco_map')
        inspect_if_panel()

    navigate_wait(x=0, y=0, z=1, speed=1, frame_id='aruco_map')
    land()

if __name__ == "__main__":
    main()
    report.close()
