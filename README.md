You can try my Demo at:
https://github.com/LukasSeratowicz/HumanPoseDetectionDemo

This Project builds on that idea implementing:
- Image to Image
- Video to Video
- Webcam Live
- Everything is embedded within Web UI using Gradio, similar to AUTOMATIC1111 for Stable Diffusion
- On top of that if I will have anough time, i will cover and add an API as well as Unity Script for example usage

Feel Free to follow for any updates!

![image](https://github.com/LukasSeratowicz/HumanPoseDetection/assets/127187274/354a7acd-654e-4c80-aa8f-7bcb1f98d75b)


Project Status:
- Image To Image <- Basic Functionality Implemented
- Video To Video <- Not Implemented
- Webcam         <- Not Implemented
- Settings       <- Implemented

Known Issues:
- There are no checks for model not loaded (crashes when trying to process image without a model loaded)
- Settings are working and are being loaded from config file, but there is a bug that does not update the color picker or checkbox look on the website (currently talking to devs)
