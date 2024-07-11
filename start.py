import threading
import gradio as gr
import numpy as np
from ultralytics import YOLO
import cv2 as cv
import torch
import os
import glob
from PIL import Image
import time

# Settings
class Color:
    r = 0
    g = 0
    b = 0
keypoints_color = Color()
keypoints_color.g = 255
keypoints_size = int(4)
line_color = Color()
line_color.b = 255
line_size = int(2)

preload_weights = True

def change_keypoints_color(red,green,blue):
    keypoints_color.r = red
    keypoints_color.g = green
    keypoints_color.b = blue
def change_line_color(red,green,blue):
    line_color.r = red
    line_color.g = green
    line_color.b = blue

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))



def apply_settings(new_keypoints_color, new_line_color, new_preload_weights):
    k_r, k_g, k_b = hex_to_rgb(new_keypoints_color)
    keypoints_color.r = k_r
    keypoints_color.g = k_g
    keypoints_color.b = k_b

    l_r, l_g, l_b = hex_to_rgb(new_line_color)
    line_color.r = l_r
    line_color.g = l_g
    line_color.b = l_b

    global preload_weights
    preload_weights = new_preload_weights

    gr.Info("Settings Changed Successfully",duration=3)
    return "Settings Changed"

# Check if running on CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'CPU'
print(f"Running on {device}")
print(f"Using OpenCV v{cv.__version__}")
if(device == 'cuda'):
    print(f"Using GPU with Cuda v{torch.version.cuda}")

### Get All Models from ./models
def find_pt_files(file_path):
    pattern = os.path.join(file_path, "*.pt")
    return glob.glob(pattern)

all_models = find_pt_files(os.path.dirname(os.path.realpath(__file__))+"\\models")
print(all_models)

### Pose Estimation AI
file_path = os.path.dirname(os.path.realpath(__file__))
model = None
KEYPOINT_PAIRS = [
    (0, 1), (0, 2), (2, 4), (1, 3), (0, 5), (0, 6), # Head
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),   # Right arm
    (6, 12), (5, 11), (6, 5), (12, 11),  # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]

def yolov8_load_model(model_name):
    if model_name is None or model_name == "":
        raise gr.Error(f"FAILED Loading Model: {model_name}", duration=5)
    global model
    print(f"Loading models//{model_name}...")
    model = YOLO("models//"+model_name)
    if preload_weights:
       load_weights()
    print(f"Model Loaded")
    print(f"Checking parameters")
    model_info = torch.load("models//"+model_name, map_location=torch.device('cpu'))
    total_params = sum(p.numel() for p in model.parameters())
    formatted_params = '{:,}'.format(total_params).replace(',', '.')
    file_size = os.path.getsize("models//"+model_name) / (1024 * 1024)
    model_info = None
    gr.Info(f"Model: {model_name} loaded successfuly", duration=3)
    return f"Model: {model_name}\nSize: {file_size:.2f} MB\nParameters: {formatted_params}"

def load_weights():
    global model
    img = "soldiers.png"
    results = model(source=img, show=False, conf=0.3, save=False)


### IMG TO IMG
def yolov8_process_image(img):
    timer_start = time.perf_counter()
    global model
    if img is None:
        raise gr.Error("No selected image found, please upload an image first 💥!", duration=5)
    results = model(source=img, show=False, conf=0.3, save=False)
    print(results) # DELETE LATER
    if isinstance(img, Image.Image):
        img = np.array(img)

    if not isinstance(img, np.ndarray):
        raise gr.Error("Image is not a NumPy array or a PIL Image", duration=5)

    image = img
    
    people_count = len(results[0].keypoints.xy)
    for person_idx in range(people_count):
        keypoints = results[0].keypoints.xy[person_idx]
        keypoints_scaled = [(int(x), int(y)) for x, y in keypoints]
        for x, y in keypoints_scaled:
            if x != 0 and y != 0:
                cv.circle(image, (x, y), int(keypoints_size), (keypoints_color.r, keypoints_color.g, keypoints_color.b), int(keypoints_size/2))
        for (start_idx, end_idx) in KEYPOINT_PAIRS:
            if start_idx < len(keypoints_scaled) and end_idx < len(keypoints_scaled):
                start_point = keypoints_scaled[start_idx]
                end_point = keypoints_scaled[end_idx]
                if start_point != (0, 0) and end_point != (0, 0):
                    cv.line(image, start_point, end_point, (line_color.r, line_color.g, line_color.b), int(line_size))

    timer_end = time.perf_counter()
    gr.Info(f"Processing Finished\nafter {timer_end-timer_start:0.4f} seconds", duration=5)
    return f"OK.\nPeople Count: {people_count}\nIt took {timer_end-timer_start:0.4f} seconds", image


### VID TO VID
def yolov8_process_video(video):
    return "NOT OK"

### WEBCAM
def yolov8_process_webcam(feed):
    return "NOT OK"


### GRADIO
tabs = ["img2img", "vid2vid", "webcam", "settings"]
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    # Github icon
    #github_icon = gr.Gallery(values=["logo\\github-mark-white.png"], interactive=False) # I have no idea how Gradio can display logos o.O
    #
    title_box = gr.Textbox(label="Human Pose Detection",value=f"Running on {device}\nUsing OpenCV v{cv.__version__} "+(f" with cuda v{torch.version.cuda}" if device == 'cuda' else "without cuda"))
    model_load_logs = gr.Textbox(label="Model Info",value=f"No Model Loaded")
    with gr.Row():
        version_selection = gr.Dropdown(choices=[os.path.basename(model) for model in all_models], label="Select Yolov8 Model")
        load_btn = gr.Button(value="Load Model")
        load_btn.click(yolov8_load_model, inputs=[version_selection], outputs=[model_load_logs])

    ### Image To Image
    with gr.Tab(tabs[0]):
        tab_name = gr.Text(value=tabs[0], visible=False)
        image = gr.Image(label="Image")
        process_btn = gr.Button(value="Process Image")
        out_text = gr.Text(value="process image to get an output", label="Output Logs")
        out_image = gr.Image(label="Output Image")
        process_btn.click(yolov8_process_image,inputs=[image],outputs=[out_text, out_image])

    ### Video To Video
    with gr.Tab(tabs[1]):
        tab_name = gr.Text(value=tabs[1], visible=False)
        video = gr.Video(label="Video") #, sources=['upload']
        video_out_text = gr.Text(value="process video to get an output", label="Output Logs")
        video_btn = gr.Button(value="Process Video")
        video_btn.click(yolov8_process_video,inputs=[video],outputs=[video_out_text])

    ### Webcam Live
    with gr.Tab(tabs[2]):
        tab_name = gr.Text(value=tabs[2], visible=False)
        webcam = gr.Video(label="Webcam", sources=['webcam']) #PLACEHOLDER #streaming=True
        webcam_out_text = gr.Text(value="process webcam to get an output", label="Output Logs")
        webcam_btn = gr.Button(value="Process Webcam")
        webcam_btn.click(yolov8_process_webcam,inputs=[webcam],outputs=[webcam_out_text])
    
    ### Settings
    with gr.Tab(tabs[3]):
        with gr.Row():
            picker_keypoints_color = gr.ColorPicker(label="Keypoints Color", value="#00FF00")
            picker_lines_color = gr.ColorPicker(label="Lines Color", value="#0000FF")
        settings_preload_weights = gr.Checkbox(label="Preload Weights on Model Change", value=preload_weights)
        settings_btn = gr.Button(value="Save Changes")
        out_text = gr.Text(value=" ", label="Output Logs")
        settings_btn.click(apply_settings,inputs=[picker_keypoints_color,picker_lines_color,settings_preload_weights],outputs=[out_text])


        #gr.themes.builder() #BROKEN
if __name__ == "__main__":
    demo.launch(show_api=False)
