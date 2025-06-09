import os
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

class VideoCaptioner:
    def __init__(self):
        # Initialize the vision-language model
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = AutoModelForCausalLM.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
    def extract_frames(self, video_path, num_frames=10):
        """Extract evenly spaced frames from the video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames
    
    def generate_caption(self, frames):
        """Generate a caption for the video using the frames."""
        # Process frames and generate caption
        inputs = self.processor(images=frames, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate caption
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            length_penalty=1.0,
            temperature=0.7
        )
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def process_video(self, video_path):
        """Process a single video and return its caption."""
        print(f"Processing video: {os.path.basename(video_path)}")
        frames = self.extract_frames(video_path)
        caption = self.generate_caption(frames)
        return caption
    
    def process_directory(self, directory_path, output_file="video_captions.json"):
        """Process all videos in a directory and save captions to a JSON file."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        captions = {}
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(directory_path, filename)
                try:
                    caption = self.process_video(video_path)
                    captions[filename] = caption
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        # Save captions to JSON file
        with open(output_file, 'w') as f:
            json.dump(captions, f, indent=4)
        
        return captions

def main():
    # Initialize the captioner
    captioner = VideoCaptioner()
    
    # Process videos in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    captions = captioner.process_directory(current_dir)
    
    # Print results
    print("\nGenerated Captions:")
    print("==================")
    for video, caption in captions.items():
        print(f"\nVideo: {video}")
        print(f"Caption: {caption}")

if __name__ == "__main__":
    main() 