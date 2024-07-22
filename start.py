import gradio as gr
from gradio_modal import Modal
import numpy as np
from ultralytics import YOLO
import cv2 as cv
import torch
import os
import glob
from PIL import Image
import time
import json

# Settings Non Editable
page_title = "HumanPoseDetection - Seratowicz"
settings_loaded = False
settings_folder = os.path.dirname(os.path.abspath(__file__))
settings_file_name = "config.json"
session_folder = os.path.dirname(os.path.abspath(__file__))
session_file_name = "session.json"
webcam_on = False
refresh_symbol = '\U0001f504'  # 🔄
file_type_list = ["jpg","png"]
frames_folder = 'frames/'
# Settings Editable
class Color:
    def __init__(self, r=0, g=0, b=0):
        self.r = r
        self.g = g
        self.b = b

    def to_dict(self):
        return {'r': self.r, 'g': self.g, 'b': self.b}
    
    @staticmethod
    def from_dict(data):
        return Color(r=data.get('r', 0), g=data.get('g', 0), b=data.get('b', 0))

keypoints_color = Color(r=0, g=255, b=0)
line_color = Color(r=0, g=0, b=255)
keypoints_size = 2
line_size = 4

preload_weights = True

confidence = 0.3

file_type = "jpg"

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

def save_settings():
    global settings_folder
    global settings_file_name
    settings = {
        'keypoints_color': keypoints_color.to_dict(),
        'keypoints_size': keypoints_size,
        'line_color': line_color.to_dict(),
        'line_size': line_size,
        'preload_weights': preload_weights,
        'confidence': confidence,
        'file_type': file_type
    }
    
    if not settings_folder:
        settings_folder = os.path.dirname(os.path.abspath(__file__)) 
    
    settings_file_path = os.path.join(settings_folder, settings_file_name)
    
    os.makedirs(settings_folder, exist_ok=True)
    
    with open(settings_file_path, 'w') as file:
        json.dump(settings, file, indent=4)

def apply_settings(new_keypoints_color, new_line_color, new_preload_weights, new_keypoints_size, new_line_size, new_confidence, new_file_type):
    global preload_weights
    global keypoints_size
    global line_size
    global confidence
    global file_type
    k_r, k_g, k_b = hex_to_rgb(new_keypoints_color)
    keypoints_color.r = k_r
    keypoints_color.g = k_g
    keypoints_color.b = k_b

    l_r, l_g, l_b = hex_to_rgb(new_line_color)
    line_color.r = l_r
    line_color.g = l_g
    line_color.b = l_b

    preload_weights = new_preload_weights

    keypoints_size = new_keypoints_size
    line_size = new_line_size

    confidence = new_confidence

    file_type = new_file_type

    save_settings()

    gr.Info("Settings Changed Successfully",duration=3)
    return "Settings Changed"

def load_settings():
    global settings_loaded
    global keypoints_color
    global keypoints_size
    global line_color
    global line_size
    global preload_weights
    global confidence
    global file_type
    settings_file_path = os.path.join(settings_folder, settings_file_name)
    
    if not os.path.isfile(settings_file_path):
        save_settings()
        
        settings_loaded = False
        return False, "No settings file found. A new settings file has been created."
    
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
        
        keypoints_color = Color.from_dict(settings.get('keypoints_color', {}))
        keypoints_size = settings.get('keypoints_size', 4)
        line_color = Color.from_dict(settings.get('line_color', {}))
        line_size = settings.get('line_size', 2)
        preload_weights = settings.get('preload_weights', True)
        confidence = settings.get('confidence', 0.3)
        file_type = settings.get('file_type', "jpg")

    settings_loaded = True
    return True, "Settings file loaded successfully.", keypoints_color, keypoints_size, line_color, line_size, preload_weights, confidence, file_type

def reload_gradio_from_settings():
    global file_type
    #print("Tab Selected")
    status = load_settings()
    #print(status)
    return gr.ColorPicker(value=f"#{keypoints_color.r:02x}{keypoints_color.g:02x}{keypoints_color.b:02x}"), gr.ColorPicker(value=f"#{line_color.r:02x}{line_color.g:02x}{line_color.b:02x}"),gr.Slider(minimum=1,maximum=16,step=1,value=keypoints_size,label="Keypoint Size"), gr.Slider(minimum=1,maximum=16,step=1,value=line_size,label="Line Size"), gr.Checkbox(label="Preload Weights on Model Change", value=preload_weights), gr.Slider(minimum=0.01,maximum=1,step=0.01,value=confidence,label="Model Confidence"), gr.Radio(value=file_type)

# SESSION
first_time = True
session_model_name = None
session_model_load_logs = None

def save_session():
    global session_folder
    global session_file_name
    global session_model_name
    global session_model_load_logs
    session = {
        'session_model_name': session_model_name,
        'session_model_load_logs': session_model_load_logs
    }
    
    if not session_folder:
        session_folder = os.path.dirname(os.path.abspath(__file__)) 
    
    session_file_name = os.path.join(session_folder, session_file_name)
    
    os.makedirs(session_folder, exist_ok=True)
    
    with open(session_file_name, 'w') as file:
        json.dump(session, file, indent=4)
    print("Successfuly saved session")


def load_session():
    global session_model_name
    global session_model_load_logs
    global session_folder
    global session_file_name

    session_file_path = os.path.join(session_folder, session_file_name)
    if not os.path.isfile(session_file_path):
        save_session()
        return None, "No Model Loaded"
    
    with open(session_file_path, 'r') as file:
        session = json.load(file)
        #print(session)
        session_model_name = session.get('session_model_name', None)
        session_model_load_logs = session.get('session_model_load_logs', "No Model Loaded")


def on_page_load():
    global session_model_name
    global session_model_load_logs
    global session_folder
    global session_file_name
    global first_time
    #print("Page Reloaded")
    if first_time:
        session_model_name = None
        session_model_load_logs = "No Model Loaded"
        file_path = os.path.join(session_folder, session_file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        first_time = False
    else:
        load_session()
        #print(f"loaded: {session_model_name} and {session_model_load_logs}")
    #status = load_settings()
    if session_model_load_logs == None or session_model_load_logs == "":
        session_model_load_logs = "No Model Loaded"
    return gr.Textbox(label="Model Info",value=session_model_load_logs), gr.Dropdown(choices=[os.path.basename(model) for model in all_models], label="Select Yolov8 Model", value=session_model_name)
    
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

def reload_models():
    global all_models
    all_models = find_pt_files(os.path.dirname(os.path.realpath(__file__))+"\\models")
    return gr.Dropdown(choices=[os.path.basename(model) for model in all_models], label="Select Yolov8 Model")

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
    global session_model_name
    global session_model_load_logs
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
    session_model_name = model_name
    session_model_load_logs = f"Model: {model_name}\nSize: {file_size:.2f} MB\nParameters: {formatted_params}"
    save_session()
    gr.Info(f"Model: {model_name} loaded successfuly", duration=3)
    return f"Model: {model_name}\nSize: {file_size:.2f} MB\nParameters: {formatted_params}"

def load_weights():
    global model
    img = "soldiers.png"
    results = model(source=img, show=False, conf=confidence, save=False)


### IMG TO IMG
def yolov8_process_image(img, print_info):
    timer_start = time.perf_counter()
    global model
    if img is None:
        raise gr.Error("No selected image found, please upload an image first 💥!", duration=5)
    if model is None:
        raise gr.Error("Please select and load model first 💥!", duration=5)
    results = model(source=img, show=False, conf=confidence, save=False)
    print(results) # DELETE LATER
    if isinstance(img, Image.Image):
        img = np.array(img)

    if not isinstance(img, np.ndarray):
        raise gr.Error("Image is not a NumPy array or a PIL Image", duration=5)

    image = img
    
    people_count = len(results[0].keypoints.xy)

    if people_count==0:
        raise gr.Error("No people found in the picture!", duration=5)
        return f"OK.\nPeople Count: {people_count}\nIt took {timer_end-timer_start:0.4f} seconds", image

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
    if print_info == True:
        gr.Info(f"Processing Finished\nafter {timer_end-timer_start:0.4f} seconds", duration=5)
    return f"OK.\nPeople Count: {people_count}\nIt took {timer_end-timer_start:0.4f} seconds", image


### VID TO VID
import subprocess
def yolov8_process_video(video, model_name):
    global frames_folder
    if video == None:
        gr.Error(f"Video is empty !", duration=3)

    # Create 'frames' folder if it doesn't exist
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    else:
        print("Normally frames would be deleted, not now")
        # Delete only frame files from the 'frames' folder if it already exists
        # print(f"Deleting old video frames from {frames_folder}")
        # for filename in os.listdir(frames_folder):
        #     del_file_path = os.path.join(frames_folder, filename)
        #     try:
        #         os.remove(del_file_path)
        #     except Exception as e:
        #         print(f"Failed to delete {del_file_path}. Reason: {e}")

    # Extract frames using ffmpeg
    # with open(temp_video_path, "wb") as f:
    #     f.write(video)

    # Extract frames using ffmpeg
    print(f"Extracting frames from {video}")
    subprocess.run(['ffmpeg', '-i', video, os.path.join(frames_folder, 'frame%d.jpg')])
    # Uncomment this after installing FFMPEG
    

    
    # Check the list of directories in 'raw_output' folder before running YOLOv8
    if not os.path.exists('raw_output'):
        os.makedirs('raw_output')
    existing_folders_before = os.listdir('raw_output')

    print(f"Loading model : {model_name}")
    model = YOLO("models//"+model_name)

    frame_file_paths = [os.path.join(frames_folder, filename) for filename in os.listdir(frames_folder) if
                        filename.endswith(f".{file_type}")]
    print(f"Processing...")
    model(source=frame_file_paths, show=False, save=True, conf=confidence, project='raw_output')
    print(f"Processing Finished")
    print(f"Creating MP4 file")
    # Check the list of directories in 'raw_output' folder after running YOLOv8
    existing_folders_after = os.listdir('raw_output')

    # Find the newly created folder
    new_folder = [folder for folder in existing_folders_after if folder not in existing_folders_before]
    if new_folder:
        new_folder_path = os.path.join('raw_output', new_folder[0])
        print(f"New folder created: {new_folder_path}")

        # Get the framerate of the original video
        # Open the video file
        video = cv.VideoCapture(file_path)

        # Get the frame rate of the video
        frame_rate = video.get(cv.CAP_PROP_FPS)

        # Release the video capture object
        video.release()

        output_video_path = os.path.join(f"{os.path.splitext(os.path.basename(file_path))[0]} [model={model_name} filetype={file_type} confidence={confidence}].mp4")

        if os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
            except Exception as e:
                print(f"Failed to delete {output_video_path}. Reason: {e}")

        # Generate MP4 video using ffmpeg
        frame_name= f"frame%d.{file_type}";
        subprocess.run(['ffmpeg', '-framerate', str(frame_rate), '-i', os.path.join(new_folder_path, frame_name), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path])

        print(f"Video generated: {output_video_path}")
        gr.Info("Successfully Crated a Video")

    else:
        print("ERROR - No new folder created")

    return f"TEST: {existing_folders_before}", output_video_path

### WEBCAM
def yolov8_process_webcam():
    global webcam_on
    webcam_on = True
    camera_id = 0
    print(f"Turning on camera id: {camera_id}")
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera id: {camera_id}")
        exit()
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    #print(f"width:{frame_width} | height:{frame_height}")
    processed_image = None
    while webcam_on:
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            gr.Error("Can't receive frame (stream end?). Exiting ...")

        #return "TESTING", frame

        status, processed_image = yolov8_process_image(frame, print_info=False)
        yield status, processed_image
    
    return "Finished Campturing Webcam", processed_image

def yolov8_process_webcam_stop():
    global webcam_on
    webcam_on = False

# CSS HERE
css = """
#small-button {
    flex: 0 1 0% !important;
    font-size: 24px !important;
    min-width: min(45px, 100%) !important;
}
"""
# TESTS GO HERE
### GRADIO
tabs = ["img2img", "vid2vid", "webcam", "settings"]
block = gr.Blocks(title=page_title, css=css).queue() #css="footer {visibility: hidden}", 
#with gr.Blocks(title=page_title, css=css) as demo:
with block as demo:
    gr.Markdown('# Human Pose Detection')

    # Github icon
    #github_icon = gr.Gallery(values=["logo\\github-mark-white.png"], interactive=False) # I have no idea how Gradio can display logos o.O
    #
    title_box = gr.Textbox(label="Human Pose Detection",show_label=False,value=f"Running on {device}\nUsing OpenCV v{cv.__version__} "+(f" with cuda v{torch.version.cuda}" if device == 'cuda' else "without cuda"))
    with gr.Accordion(label='Step 1: Choose a Model', open=True):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_selection = gr.Dropdown(choices=[os.path.basename(model) for model in all_models], label="Select Yolov8 Model")
                        reload_btn = gr.Button(value=refresh_symbol, elem_id="small-button")
                load_btn = gr.Button(value="Load Model")
            with gr.Accordion(label="Model Info", open=True):
                model_load_logs = gr.Textbox(show_label=False, value=f"No Model Loaded")
        reload_btn.click(reload_models, inputs=[], outputs=[model_selection])
        load_btn.click(yolov8_load_model, inputs=[model_selection], outputs=[model_load_logs])
    with gr.Accordion(label='Step 2: Choose a mode and run it', open=True):
        ### Image To Image
        with gr.Tab(tabs[0]):
            with gr.Group():
                tab_name = gr.Text(value=tabs[0], visible=False)
                image = gr.Image(label="Image")
                process_btn = gr.Button(value="Process Image")
            with gr.Group():
                out_image = gr.Image(label="Output Image")
                work_around_false = gr.Checkbox(label="If you see this, there is something wrong !", value=False, visible=False)
                out_text = gr.Text(value="process image to get an output", label="Output Logs")
            process_btn.click(yolov8_process_image,inputs=[image, work_around_false],outputs=[out_text, out_image])

        ### Video To Video
        with gr.Tab(tabs[1]):
            with gr.Group():
                tab_name = gr.Text(value=tabs[1], visible=False)
                video = gr.Video(label="Video") #, sources=['upload']
                video_btn = gr.Button(value="Process Video")
            with gr.Group():
                video_output = gr.Video(label="Video Output")
                video_out_text = gr.Text(value="process video to get an output", label="Output Logs")


            video_btn.click(yolov8_process_video,inputs=[video, model_selection],outputs=[video_out_text, video_output])

        ### Webcam Live
        with gr.Tab(tabs[2]) as webcam_tab:
            with gr.Group():
                webcam_out_img = gr.Image(value="process webcam to get an output", label="Output Image")
                webcam_btn_start = gr.Button(value="Start")
                webcam_btn_stop = gr.Button(value="Stop")
            with gr.Group():
                webcam_out_text = gr.Text(value="process webcam to get an output", label="Output Logs")

            webcam_btn_start.click(yolov8_process_webcam, inputs=[], outputs=[webcam_out_text, webcam_out_img]) 
            webcam_btn_stop.click(yolov8_process_webcam_stop, inputs=[], outputs=[]) 

        ### Settings
        with gr.Tab(tabs[3]) as settings_tab:
            with gr.Accordion(label='Global Settings', open=True):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            picker_keypoints_color = gr.ColorPicker(label="Keypoints Color", value="#00FF00")
                            slider_keypoints_size = gr.Slider(minimum=1,maximum=16,step=1,value=keypoints_size,label="Keypoint Size")
                    with gr.Column():
                        with gr.Group():
                            picker_lines_color = gr.ColorPicker(label="Lines Color", value="#0000FF")
                            slider_lines_size = gr.Slider(minimum=1,maximum=16,step=1,value=line_size,label="Line Size")
                with gr.Group():
                    settings_preload_weights = gr.Checkbox(label="Preload Weights on Model Change", value=preload_weights)
                    settings_confidence = gr.Slider(minimum=0.01,maximum=1,step=0.01,value=confidence,label="Model Confidence")
            
            with gr.Accordion(label='Vid2vid Settings', open=True):
                with gr.Row():
                    settings_file_type = gr.Radio(file_type_list, value=file_type, label="File type", interactive=True, info="png: slower, bigger file, higher accuracity, higher quality")
            settings_btn = gr.Button(value="Save Changes")
            out_text = gr.Text(value=" ", label="Output Logs")

            show_btn = gr.Button("View Changelog")

            with Modal(visible=False) as modal:
                gr.Markdown("# Changelog")
                # with gr.Accordion(label='# Version 0.3 - Ongoing', open=True):
                #     with gr.Group():
                #         with gr.Accordion(label='# Version 0.3.0 - Ongoing', open=True):
                #             gr.Markdown("- come back here later")
                with gr.Accordion(label='# Version 0.2 - Newest version released on 19/07/2024', open=True):
                    with gr.Group():
                        with gr.Accordion(label='# Version 0.2.7 - 19/07/2024', open=True):
                            gr.Markdown("- Changelog Implemented")
                            gr.Markdown("- Files Clean Up")
                            gr.Markdown("- UI changes - Grouping of elements added")
                        with gr.Accordion(label='# Version 0.2.6 - 16/07/2024', open=False):
                            gr.Markdown("- Vid2vid basic version implemented")
                            gr.Markdown("- UI changes")
                        with gr.Accordion(label='# Version 0.2.5 - 15/07/2024', open=False):
                            gr.Markdown("- CSS now works. Changed Button Look")
                            gr.Markdown("- Session File added to save basic stuff")
                            gr.Markdown("- Refresh Model List added")
                            gr.Markdown("- Webcam now works using opencv")
                        with gr.Accordion(label='# Version 0.2.4 - 15/07/2024', open=False):
                            gr.Markdown("- Images Clean Up")
                            gr.Markdown("- Webcam now works with snapshot button")
                            gr.Markdown("- Gradio elements now refresh correctly on page reload")
                        with gr.Accordion(label='# Version 0.2.3 - 15/07/2024', open=False):
                            gr.Markdown("- Requirements.txt added")
                            gr.Markdown("- Settings implemented")
                        with gr.Accordion(label='# Version 0.2.2 - 12/07/2024', open=False):
                            gr.Markdown("- vid2vid and webcam ui placeholders")
                            gr.Markdown("- Clean Up of code and pictures")
                        with gr.Accordion(label='# Version 0.2.1 - 11/07/2024', open=False):
                            gr.Markdown("- Model Loading Progress Bar added")
                            gr.Markdown("- Model Info Text Box added")
                            gr.Markdown("- Various checks for wrong input added")
                with gr.Accordion(label='# Version 0.1 - Newest version released on 11/07/2024', open=False):
                    gr.Markdown("- Basic UI Implemented")
                    gr.Markdown("- Loading yolov8 model added")
                    gr.Markdown("- Img2Img basic functionality added")
                    gr.Markdown("- Settings added")
                    # gr.Textbox(show_label=False, 
                    #            value="""- Basic UI Implemented\n- Loading yolov8 model added\n- Img2Img basic functionality added
                    #            """)
            show_btn.click(lambda: Modal(visible=True), None, modal)


            settings_btn.click(apply_settings,inputs=[picker_keypoints_color,picker_lines_color,settings_preload_weights,slider_keypoints_size,slider_lines_size, settings_confidence, settings_file_type],outputs=[out_text])
            #demo.load(load_settings, inputs=[], outputs=[out_text])
            settings_tab.select(reload_gradio_from_settings,inputs=[],outputs=[picker_keypoints_color,picker_lines_color, slider_keypoints_size, slider_lines_size, settings_preload_weights, settings_confidence, settings_file_type])
            #gr.themes.builder() #BROKEN
    #demo.Interface.tabs[-1].selected = load_settings
    #settings_tab.
    demo.load(on_page_load, inputs=[], outputs=[model_load_logs, model_selection])
    #print(demo.blocks)
# if __name__ == "__main__":
#     status = load_settings()
#     print(status)
#     demo.launch(share=False)


status = load_settings()
print(status)
block.queue().launch(server_name='127.0.0.1',share=False)
