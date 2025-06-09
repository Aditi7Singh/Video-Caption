# Video Captioning with LLM/VLM

This script uses the BLIP-2 model to automatically generate captions for videos. It extracts frames from videos and uses a vision-language model to generate descriptive captions.

## Features

- Processes multiple video formats (mp4, avi, mov, mkv)
- Extracts evenly spaced frames for better context
- Uses BLIP-2 model for high-quality caption generation
- Saves results in JSON format
- GPU acceleration support

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your videos in the same directory as the script
2. Run the script:
```bash
python video_captioner.py
```

The script will:
- Process all videos in the directory
- Generate captions for each video
- Save the results in `video_captions.json`
- Display the captions in the console

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- See requirements.txt for full list of dependencies

## Notes

- The script uses the BLIP-2 model which requires significant computational resources
- Processing time depends on video length and available hardware
- For best results, ensure videos are clear and well-lit 