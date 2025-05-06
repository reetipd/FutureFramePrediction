# import cv2
# from ultralytics import YOLO
# import numpy as np
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader

# # Initialize YOLOv5 model (replace with your specific model and path)
# model = YOLO("./data/traffic/best_1.pt")
# model.verbose = False

# def extract_frames(video_path, seq_length=10, resize_width=64, resize_height=64):
#     # Open the video file
#     video = cv2.VideoCapture(video_path)
#     frames = []
#     frame_count = 0

#     while video.isOpened():
#         ret, frame = video.read()
#         if not ret:
#             break

#         # Convert the frame to grayscale
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Resize the frame to the desired dimensions and normalize pixel values
#         resized_frame = cv2.resize(gray_frame, (resize_width, resize_height)) / 255.0
        
#         # Append the resized frame
#         frames.append(resized_frame)
#         frame_count += 1
#         # print(f"Frame {frame_count}: Grayscale and resized - Shape: {resized_frame.shape}")

#     video.release()
#     # print(f"Total frames extracted: {frame_count}")
#     return frames

# # Create sequences of frames (T=10) for training the model
# def preprocess_frames(frames, seq_length=10, resize_shape=(64, 64)):
#     processed_frames = []
#     sequence_count = 0  # To keep track of the number of sequences

#     # Create a sequence of frames (T = seq_length)
#     for i in range(len(frames) - seq_length):
#         frame_sequence = []

#         for j in range(seq_length):
#             frame = frames[i + j]
#             frame_tensor = torch.tensor(frame).float()  # Convert to tensor

#             # Log the individual frame's tensor conversion
#             # print(f"Processing Sequence {sequence_count + 1}: Frame {i + j} -> Tensor Shape: {frame_tensor.shape}")

#             # Stack frames to create a sequence
#             frame_sequence.append(frame_tensor)

#         # Stack the sequence of frames to make it a tensor
#         processed_frames.append(torch.stack(frame_sequence))
#         sequence_count += 1

#     # print(f"Total sequences processed: {sequence_count}")
#     return processed_frames


# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import cv2


# class TrafficVideoDataset(Dataset):
#     def __init__(self, frames, seq_length=10, n_frames_input=10, n_frames_output=10, resize_shape=(64, 64)):
#         self.frames = frames
#         self.seq_length = seq_length
#         self.resize_shape = resize_shape
#         self.n_frames_input = n_frames_input if n_frames_input is not None else 10  # default value
#         self.n_frames_output = n_frames_output if n_frames_output is not None else 5  # default value
#         self.transform = transforms.Compose([
#             transforms.Resize(resize_shape),  # Resize directly on tensor (no need for PIL conversion)
#             transforms.Grayscale(num_output_channels=1),  # Convert to grayscale, no need for PIL
#             transforms.ToTensor(),  # Converts to tensor (if not already a tensor)
#         ])


#     def __len__(self):
#         return len(self.frames) - self.seq_length  # Return number of valid sequences

#     # def __getitem__(self, index):
#     #     # Define the sequence length (adjust this as needed)
#     #     input_seq_length = self.n_frames_input
#     #     output_seq_length = self.n_frames_output
#     #     total_length = input_seq_length + output_seq_length

#     #     # Fetch frames from the dataset
#     #     frames = [self.frames[index + i] for i in range(total_length)]

#     #     # Separate the frames into input and output
#     #     input_frames = frames[:input_seq_length]
#     #     print(input_frames.shape)  # Check the number of channels
#     #     output_frames = frames[input_seq_length:]

#     #     # Apply transformations to each frame individually, not the stack
#     #     input_frames = [self.transform(frame) for frame in input_frames]
#     #     print(f"Input frames transformed shape (should be [1, H, W]): {[frame.shape for frame in input_frames]}")

#     #     output_frames = [self.transform(frame) for frame in output_frames]
#     #     print(f"Output frames transformed shape (should be [1, H, W]): {[frame.shape for frame in output_frames]}")

#     #     # Convert frames to tensors and stack them
#     #     input_tensor = torch.stack(input_frames).float()  # Stack input frames into a tensor
#     #     print(f"Input tensor shape after stacking: {input_tensor.shape}")

#     #     output_tensor = torch.stack(output_frames).float() if output_frames else torch.tensor([])

#     #     # Print the output tensor shape to verify it as well
#     #     if output_tensor.numel() > 0:
#     #         print(f"Output tensor shape after stacking: {output_tensor.shape}")
#     #     else:
#     #         print("Output tensor is empty.")

#     #     # Ensure correct dimensions, e.g., [batch_size, channels=1, height, width]
#     #     return input_tensor, output_tensor



# def load_data(batch_size, val_batch_size, data_root, num_workers, seq_length=10, resize_shape=(64, 64)):

#     video_path = f"{data_root}/traffic/video.mp4"
#     # Step 1: Extract frames and detections from video
#     frames = extract_frames(video_path)

#     print(f"Done with extracting frames from video", len(frames))

#     # Step 2: Preprocess the frames and detections (called here)
#     processed_frames = preprocess_frames(frames, seq_length=seq_length, resize_shape=resize_shape)

#     # Step 3: Create the dataset and dataloaders
#     train_set = TrafficVideoDataset(frames=processed_frames, seq_length=seq_length, resize_shape=resize_shape)
#     test_set = TrafficVideoDataset(frames=processed_frames, seq_length=seq_length, resize_shape=resize_shape)

#     # Create DataLoader for batching during training and testing
#     dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
#     dataloader_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

#     return dataloader_train, None, dataloader_test, 0, 1

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from ultralytics import YOLO

def enhance_frame(frame):
    """Apply additional enhancements to improve frame quality"""
    # Enhance contrast
    frame = cv2.equalizeHist(frame)
    
    # Optional: Apply slight Gaussian blur to reduce noise
    # frame = cv2.GaussianBlur(frame, (3,3), 0)
    
    # Optional: Sharpen
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
        frame = cv2.resize(gray_frame, (512, 512))  # Changed from 128x128 to 64x64
        # frame = enhance_frame(frame) 
        
        # Normalize to [0,1]
        frame = frame / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    frames = np.array(frames, dtype=np.float32)  # Shape: [N, H, W]
    frames = frames[:, np.newaxis, :, :]  # Shape: [N, C=1, H, W]
    
    print(f"Processed frames shape: {frames.shape}")
    return frames

# def process_video(video_path, sample_rate=3):  # sample_rate=3 means take every 3rd frame
#     print(f"Loading video from: {video_path}")
    
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     frame_count = 0
    
#     original_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     effective_fps = original_fps // sample_rate
#     print(f"Original FPS: {original_fps}")
#     print(f"Sampling every {sample_rate} frames -> Effective FPS: {effective_fps}")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Only process every Nth frame
#         if frame_count % sample_rate == 0:
#             # Convert BGR to grayscale
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             gray_frame = clahe.apply(frame)
#             gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 0)
            
#             # Resize to match model's expected input size
#             frame = cv2.resize(gray_frame, (512, 512))
            
#             # Normalize to [0,1]
#             frame = frame / 255.0
            
#             frames.append(frame)
        
#         frame_count += 1
    
#     cap.release()
    
#     frames = np.array(frames, dtype=np.float32)  # Shape: [N, H, W]
#     frames = frames[:, np.newaxis, :, :]  # Shape: [N, C=1, H, W]
    
#     print(f"Total original frames: {frame_count}")
#     print(f"Sampled frames: {len(frames)}")
#     print(f"Processed frames shape: {frames.shape}")
#     return frames


import cv2
import numpy as np
from ultralytics import YOLO

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
    # Directly transpose to [N, C, H, W] format (channels first for PyTorch)
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
        # Get sequences from the dataset
        input_sequence = self.data[index:index + self.input_frames]
        target_sequence = self.data[index + self.input_frames:index + self.input_frames + self.output_frames]
        
        # Ensure we're not accidentally using the same frames for input and target
        assert not torch.equal(input_sequence[-1], target_sequence[0]), "Input and target sequences overlap"
        
        # No dimension adjustment needed now - data should already be [T, C, H, W]
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