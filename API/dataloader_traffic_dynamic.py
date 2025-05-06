import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os

def enhance_frame(frame):
    """Apply additional enhancements to improve frame quality"""
    frame = cv2.equalizeHist(frame)

    kernel = np.array([[-1,-1,-1], 
                      [-1, 9,-1],
                      [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    
    return frame

def process_video(video_path):
    print(f"Loading video from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_frame = clahe.apply(frame)
        gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 0)
        # Resize to 64x64 to match in_shape
        frame = cv2.resize(gray_frame, (256, 256))  
        # frame = enhance_frame(frame) 
        
        # Normalize to [0,1]
        frame = frame / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    frames = np.array(frames, dtype=np.float32) 
    frames = frames[:, np.newaxis, :, :]  
    
    print(f"Processed frames shape: {frames.shape}")
    return frames

class TrafficVideoDataset(Dataset):
    def __init__(self, data, input_frames=10, output_frames=10, fps=30):  
        super(TrafficVideoDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_frames = 2 * fps    
        self.output_frames = 2 * fps   
        self.skip_frames = 4 * fps   
                
        # Add mean and std attributes
        self.mean = torch.mean(self.data).item()
        self.std = torch.std(self.data).item()
        
        print(f"Dataset initialized with shape: {self.data.shape}")
        print(f"Dataset mean: {self.mean}, std: {self.std}")
        
    def __len__(self):
        # Need enough frames for input + skip + output
        total_frames_needed = self.input_frames + self.skip_frames + self.output_frames
        return max(0, len(self.data) - total_frames_needed)
    
    def __getitem__(self, index):
        # Get first 20 seconds for input
        input_sequence = self.data[index:index + self.input_frames]
        
        # Skip 40 seconds and get next 20 seconds for prediction
        target_start = index + self.input_frames + self.skip_frames
        target_sequence = self.data[target_start:target_start + self.output_frames]
        
        # Convert to 3 channels
        input_sequence = input_sequence.repeat(1, 3, 1, 1)
        target_sequence = target_sequence.repeat(1, 3, 1, 1)
        
        if index == 0:  
            print(f"Input shape: {input_sequence.shape}")
            print(f"Target shape: {target_sequence.shape}")
        
        return input_sequence, target_sequence

def load_data(batch_size, val_batch_size, data_root, num_workers, fps=30):
    video_path = os.path.join(data_root, 'traffic/video.mp4')
    
    frames = process_video(video_path)
    print(f"Total frames in video: {len(frames)}")
    
    min_frames_needed = (10) * fps  
    if len(frames) < min_frames_needed:
        print(f"Warning: Video should be at least 40 seconds long. Current frames: {len(frames)}")
    
    # Split data
    train_size = int(0.7 * len(frames))
    val_size = int(0.15 * len(frames))
    test_size = len(frames) - train_size - val_size
    
    train_frames = frames[:train_size]
    val_frames = frames[train_size:train_size + val_size]
    test_frames = frames[train_size + val_size:]
    
    # Create datasets
    train_dataset = TrafficVideoDataset(train_frames, fps=fps)
    val_dataset = TrafficVideoDataset(val_frames, fps=fps)
    test_dataset = TrafficVideoDataset(test_frames, fps=fps)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, 0, 1