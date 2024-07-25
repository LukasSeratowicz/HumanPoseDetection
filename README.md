
# Human Pose Detection

Welcome to the Human Pose Detection project! This project makes use of YoloV8 Pose model. The system offers a user-friendly web interface powered by Gradio, enabling seamless interaction with the model. The project supports multiple input formats, including images, videos, and live webcam feeds.

## About

This Repository is a work for my Engineering Degree. A paper on it will be written in the near future, as I add features and begin my research on optimisations and works of YoloV8 model.

## Features

- **Image to Image**: Detects and analyzes poses in static images.
- **Video to Video**: Processes video files to identify and track poses throughout the footage.
- **Webcam Live**: Real-time pose detection using your webcam.
- **Web UI**: Intuitive interface built with Gradio, akin to AUTOMATIC1111 for Stable Diffusion.
- **API and Unity Script**: Future plans include API integration and Unity script examples for practical use cases.

## Project Status

- **Image To Image**: Basic functionality implemented.
- **Video To Video**: Basic functionality implemented.
- **Webcam**: Basic functionality implemented (needs more testing).
- **Settings**: Implemented.

## Screenshots

| ![Screenshot4](https://github.com/user-attachments/assets/e55a966d-2fdf-4fb2-91d9-272f953f62f9) | ![Screenshot3](https://github.com/user-attachments/assets/c2632ef7-97e1-41b4-9ee8-f32db6c21977) |
|-------------------------------------|-------------------------------------|
| ![Screenshot2](https://github.com/user-attachments/assets/1323947d-9b37-4728-8ce1-9dd96df40ce8)  | ![Screenshot1](https://github.com/user-attachments/assets/9d5de04a-8b75-4739-8a37-50231cc8e8f9) |
| ![preview6](https://github.com/user-attachments/assets/25d604e8-3f93-4bbc-824d-1e553232a7cf) | ![preview5](https://github.com/user-attachments/assets/26f98fb2-3c8d-4c27-bb44-9ed63cc6f3e1) |


## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/LukasSeratowicz/HumanPoseDetection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd HumanPoseDetection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python start.py
   ```
   
## Usage

1. Check console for address. For example:
   ```bash
   Running on local URL:  http://127.0.0.1:7860
   ```
2. Copy and paste the adress into your browser
   
## Changelog

### Version 0.3 - Ongoing

#### Version 0.3.2 - Released on 25/07/2024
- General - Updated Gradio to 4.39.0

#### Version 0.3.1 - Released on 24/07/2024
- Changelog - now opens newest sub pages automatically on first open
- vid2vid - videos now don't scale like crazy in UI (if it's not vertical, still looking for better solution)
- Removed FFMPEG testing button (oops)

#### Version 0.3.0 - Released on 23/07/2024
- Keypoints now draw over Lines for better visibility
- ffmpeg is now built-in (no need to install it locally)
- vid2vid - old frames now delete properly
- vid2vid - fps is now checked properly (no more 30 fps locked)
- vid2vid - output video name changed from placeholder
- vid2vid - css preparations
- vid2vid - other small bug fixes

### Version 0.2 - Released on 23/07/2024

#### Version 0.2.9 - 23/07/2024
##### Version 0.2.9.5 - 23/07/2024
- Unsaved changes modal - colored squares added to color changes
- Unsaved changes modal - now displays list of changes made
- Unsaved changes modal - many bug fixes

##### Version 0.2.9.0 - 22/07/2024
- Unsaved changes modal css added
- Unsaved changes modal basic functionality

#### Version 0.2.8 - 22/07/2024
- People Counter now works properly
- Keypoints and Lines now Scale with Image
- Step 1 layout change

#### Version 0.2.7 - 19/07/2024
- Changelog Implemented
- Files Clean Up
- UI changes - Grouping of elements added

#### Version 0.2.6 - 16/07/2024
- Vid2vid basic version implemented
- UI changes

#### Version 0.2.5 - 15/07/2024
- CSS now works. Changed Button Look
- Session File added to save basic stuff
- Refresh Model List added
- Webcam now works using opencv

#### Version 0.2.4 - 15/07/2024
- Images Clean Up
- Webcam now works with snapshot button
- Gradio elements now refresh correctly on page reload

#### Version 0.2.3 - 15/07/2024
- Requirements.txt added
- Settings implemented

#### Version 0.2.2 - 12/07/2024
- vid2vid and webcam ui placeholders
- Clean Up of code and pictures

#### Version 0.2.1 - 11/07/2024
- Model Loading Progress Bar added
- Model Info Text Box added
- Various checks for wrong input added

### Version 0.1 - Released on 11/07/2024
- Basic UI Implemented
- Loading yolov8 model added
- Img2Img basic functionality added
- Settings added


