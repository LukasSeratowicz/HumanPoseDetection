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
- Video To Video <- Basic Functionality Implemented (needs more testing)
- Webcam         <- Basic Functionality Implemented (needs more testing)
- Settings       <- Implemented

Known Issues:
- Gradio has trouble processing live webcam footage, might have to add a seperate button that will make a new window using openCV (for now, start stop button instead + using opencv webcam instead of gradio)
- Settings Tab loads settings from config file each time tab is opened instead of on page load (will fix that much later on)
- Do we want to make it local or deployable is the big question here. Right now it will be local only, we'll see in the future
