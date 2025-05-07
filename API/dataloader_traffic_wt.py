import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from ultralytics import YOLO


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
        frame = cv2.resize(gray_frame, (512, 512))  
        
        # Normalize to [0,1]
        frame = frame / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    frames = np.array(frames, dtype=np.float32)  
    frames = frames[:, np.newaxis, :, :] 
    
    print(f"Processed frames shape: {frames.shape}")
    return frames


def process_video_with_yolo(video_path, yolo_model_path):
    """
    Process video using YOLO detection to track vehicles.
    Combines every 3 frames into 1 frame using mask-based blending.
    """
    cap = cv2.VideoCapture(video_path)
    output_dir = "./viss"
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    frame_count = 0
    window_size = 3
    frame_buffer = []
    
    # Load YOLO model
    model = YOLO(yolo_model_path)
    model.verbose = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        # Convert BGR to RGB (OpenCV uses BGR, PyTorch expects RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = cv2.resize(frame, (512, 512))
        
        # YOLO detection (vehicle classes: car, motorcycle, bus, truck)
        results = model.predict(frame, classes=[3, 5, 8], conf=0.4)
        boxes = results[0].boxes

        frame_buffer.append((processed_frame.astype(np.float32), boxes))

        if len(frame_buffer) >= window_size:
            base_frame = frame_buffer[-1][0].copy()
            mask_accumulator = np.zeros((512, 512, 3), dtype=np.float32)
            weight_accumulator = np.zeros((512, 512, 3), dtype=np.float32)

            # Blending frames using mask-based approach
            for i, (past_frame, bboxes) in enumerate(reversed(frame_buffer[:-1])):
                weight = 0.9 ** (i + 1)
                mask = np.zeros((512, 512, 3), dtype=np.float32)
                
                for box in bboxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = int(x1 * 512 / orig_w)
                    y1 = int(y1 * 512 / orig_h)
                    x2 = int(x2 * 512 / orig_w)
                    y2 = int(y2 * 512 / orig_h)
                    x1, x2 = np.clip([x1, x2], 0, 512)
                    y1, y2 = np.clip([y1, y2], 0, 512)

                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2, :] = 1

                mask_accumulator += past_frame * mask * weight
                weight_accumulator += mask * weight

            # Normalize blended regions to avoid division by zero
            blended_regions = np.where(weight_accumulator > 0,
                                       mask_accumulator / weight_accumulator,
                                       base_frame)

            # Combine base frame with blended vehicle regions
            combined_frame = np.where(weight_accumulator > 0,
                                      blended_regions,
                                      base_frame)

            combined_frame = np.clip(combined_frame / 255.0, 0, 1)
            # Save the frame
            output_combined_path = os.path.join(output_dir, f'aggregated_{frame_count}_combined.png')
            # Convert to BGR for saving with OpenCV
            save_frame = (combined_frame * 255).astype(np.uint8)
            save_frame = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_combined_path, save_frame)
            
            frames.append(combined_frame)
            

            frame_buffer = []

        frame_count += 1

    cap.release()

    frames = np.array(frames, dtype=np.float32)
    frames = frames.transpose(0, 3, 1, 2)  # Shape: [N, C=3, H, W]

    print(f"Total original frames: {frame_count}")
    print(f"Processed frames shape: {frames.shape}")
    return frames


class TrafficVideoDataset(Dataset):
    def __init__(self, data, input_frames=20, output_frames=20):
        super(TrafficVideoDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_frames = input_frames
        self.output_frames = output_frames
        
        # Add mean and std attributes
        self.mean = torch.mean(self.data).item()
        self.std = torch.std(self.data).item()
        
        print(f"Dataset initialized with shape: {self.data.shape}")
        print(f"Dataset mean: {self.mean}, std: {self.std}")
        
    def __len__(self):
        return max(0, len(self.data) - (self.input_frames + self.output_frames))
    
    def __getitem__(self, index):
        input_sequence = self.data[index:index + self.input_frames]
        target_sequence = self.data[index + self.input_frames:index + self.input_frames + self.output_frames]
        
        assert not torch.equal(input_sequence[-1], target_sequence[0]), "Input and target sequences overlap"
        
        return input_sequence, target_sequence

def load_data(batch_size, val_batch_size, data_root, num_workers):
    video_path = os.path.join(data_root, 'traffic/video.mp4')
    yolo_model_path = os.path.join(data_root, 'traffic/best_1.pt')
    
    # Process video with YOLO-based detection
    frames = process_video_with_yolo(video_path, yolo_model_path)
    
    # Split data
    train_size = int(0.7 * len(frames))
    val_size = int(0.15 * len(frames))
    test_size = len(frames) - train_size - val_size
    
    train_frames = frames[:train_size]
    val_frames = frames[train_size:train_size + val_size]
    test_frames = frames[train_size + val_size:]
    
    # Create datasets
    train_dataset = TrafficVideoDataset(train_frames, input_frames=10, output_frames=10)
    val_dataset = TrafficVideoDataset(val_frames, input_frames=10, output_frames=10)
    test_dataset = TrafficVideoDataset(test_frames, input_frames=10, output_frames=10)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, 0, 1

if __name__ == "__main__":
    train_loader, val_loader, test_loader, mean, std = load_data(
        batch_size=4, val_batch_size=4, data_root='./data', num_workers=0
    )