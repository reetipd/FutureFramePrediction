import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import numpy as np
from ultralytics import YOLO

def process_video(
    yolo_model_path: str,
    video_path: str,
    window_size: int = 30,
    iou_thresh: float = 0.3,
    output_dir: str = "./visualizations"
):
    """
    For each window of `window_size` frames, produces one composite image
    showing every vehicle (crisp, full-color) overlaid on the true background.
    """
    model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    mask_dir = os.path.join(output_dir, "masks")
    img_dir = os.path.join(output_dir, "composites")
    inter_dir = os.path.join(output_dir, "inter_masks")
    os.makedirs(mask_dir,  exist_ok=True)
    os.makedirs(img_dir,   exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    def compute_iou(b1, b2):
        x1, y1, x2, y2   = b1
        x1p, y1p, x2p, y2p = b2
        xi1, yi1 = max(x1, x1p), max(y1, y1p)
        xi2, yi2 = min(x2, x2p), min(y2, y2p)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        a1 = (x2 - x1) * (y2 - y1)
        a2 = (x2p - x1p) * (y2p - y1p)
        return inter / (a1 + a2 - inter)

    frames, detections = [], []
    batch = 0
    model_input_frames = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        res = model.predict(frame, classes=[3,5,8], conf=0.4)
        boxes = [
            (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
             int(b.xyxy[0][2]), int(b.xyxy[0][3]))
            for b in res[0].boxes
        ]
        detections.append(boxes)

        if len(frames) == window_size:
            batch += 1
            h, w  = frames[0].shape[:2]
            
           
            all_frames = np.array(frames)
            background = np.median(all_frames, axis=0).astype(np.uint8)
            
            full_masks = []
            prev_boxes = detections[0]
            prev_mask  = np.zeros((h, w), dtype=np.uint8)

            for i in range(1, window_size):
                curr_boxes    = detections[i]
                inter_mask    = np.zeros((h, w), dtype=np.uint8)
                curr_box_mask = np.zeros((h, w), dtype=np.uint8)

                for pb in prev_boxes:
                    for cb in curr_boxes:
                        if compute_iou(pb, cb) > iou_thresh:
                            xi1, yi1 = max(pb[0], cb[0]), max(pb[1], cb[1])
                            xi2, yi2 = min(pb[2], cb[2]), min(pb[3], cb[3])
                            prev_mask[pb[1]:pb[3], pb[0]:pb[2]] = 0
                        else:
                            prev_mask[pb[1]:pb[3], pb[0]:pb[2]] = 255

                for (x1,y1,x2,y2) in curr_boxes:
                    curr_box_mask[y1:y2, x1:x2] = 255

                step_mask = cv2.bitwise_or(prev_mask, curr_box_mask)
                full_masks.append(step_mask.copy())

                prev_boxes = curr_boxes
                prev_mask  = step_mask

            composite = np.zeros_like(frames[0])
            for i, mask in enumerate(full_masks, start=1):
                
                cv2.copyTo(frames[i], mask, composite)
         
                cv2.imwrite(
                    os.path.join(inter_dir, f"batch{batch:03d}_step{i:03d}.png"),
                    mask
                )

            gray      = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
            ys, xs    = np.where(gray > 0)
            final_img = background.copy()
            final_img[ys, xs] = composite[ys, xs]

            cv2.imwrite(os.path.join(img_dir,  f"comp_batch_{batch:03d}.png"),
                        final_img)
            
            # final_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model_frame = cv2.resize(final_img, (512, 512))
            model_frame = np.clip(model_frame / 255.0, 0, 1)
            model_input_frames.append(model_frame)
          
            print(f"[Batch {batch:03d}] composite → {img_dir}, mask → {mask_dir}")

            frames.clear()
            detections.clear()

    cap.release()

    model_input_frames = np.array(model_input_frames, dtype=np.float32)
    model_input_frames = model_input_frames.transpose(0, 3, 1, 2)
    print(f"Returning {len(model_input_frames)} frames for model input with shape {model_input_frames.shape}")
    return model_input_frames



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
        print("The index is: ", index)
        # Get sequences from the dataset
        input_sequence = self.data[index:index + self.input_frames]

        # changes made to predict only 2 frames with overlapping
        target_sequence = self.data[index + self.input_frames - (self.input_frames - self.output_frames):index + self.input_frames + self.output_frames]

        #TODO asserting no overlapping but we want overlapping
        assert not torch.equal(input_sequence[-1], target_sequence[0]), "Input and target sequences overlap"
        
        return input_sequence, target_sequence

def load_data(batch_size, val_batch_size, data_root, num_workers):
    video_path = os.path.join(data_root, 'traffic/video.mp4')

    # TODO either retrain the YOLO model for the blurr frames or make frames more readable
    yolo_model_path = os.path.join(data_root, 'traffic/best_1.pt')
    
    # Process video with YOLO-based detection
    frames = process_video(yolo_model_path, video_path)
    
    print(f"DEBUG: Total frames processed: {len(frames)}")
    
    # Split data
    train_size = int(0.7 * len(frames))
    val_size = int(0.15 * len(frames))
    test_size = len(frames) - train_size - val_size
    
    train_frames = frames[:train_size]
    val_frames = frames[train_size:train_size + val_size]
    test_frames = frames[train_size + val_size:]
    
    print(f"DEBUG: Split sizes - Train: {len(train_frames)}, Val: {len(val_frames)}, Test: {len(test_frames)}")
    
    # Create datasets
    train_dataset = TrafficVideoDataset(train_frames, input_frames=10, output_frames=2)
    val_dataset = TrafficVideoDataset(val_frames, input_frames=10, output_frames=2)
    test_dataset = TrafficVideoDataset(test_frames, input_frames=10, output_frames=2)
    
    print(f"DEBUG: Dataset lengths - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
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
    
    print(f"DEBUG: DataLoader batch counts - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, 0, 1

# if __name__ == "__main__":
#     train_loader, val_loader, test_loader, mean, std = load_data(
#         batch_size=4, val_batch_size=4, data_root='./data', num_workers=0
#     )