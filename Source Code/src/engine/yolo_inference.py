import argparse 
import cv2
import numpy as np 
import os 

import torch 
from torchvision import transforms

from inference import infer
from yolo_utils import prepare_yolo, rescale_boxes, non_max_suppression

import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--video_file', type=str, help='Test video file')
    # Generate args
    args = parser.parse_args()
    return args


def get_bbox(yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):

  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

  with torch.no_grad():
    detections = yolo_model(image_yolo)
    nms_det  = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
    det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))
  
  bboxes = []
  for x1, y1, x2, y2, _, _, cls_pred in det:
    if cls_pred == 0:  # checking if predicted_class = persons. 
      x1 = int(min(image_context.shape[1], max(0, x1)))
      x2 = int(min(image_context.shape[1], max(x1, x2)))
      y1 = int(min(image_context.shape[0], max(15, y1)))
      y2 = int(min(image_context.shape[0], max(y1, y2)))
      bboxes.append([x1, y1, x2, y2])
  return np.array(bboxes)



def yolo_video(video_file, result_path, model_path, context_norm, body_norm, ind2cat, args, skip_frames=9):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    yolo = prepare_yolo(model_path).to(device).eval()
    thresholds = torch.FloatTensor(np.load(os.path.join(model_path, 'val_thresholds.npy'))).to(device)
    
    # Load models
    model_context = torch.load(os.path.join(model_path,'model_context1.pth'), weights_only=False).to(device).eval()
    model_body = torch.load(os.path.join(model_path,'model_body1.pth'), weights_only=False).to(device).eval()
    emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth'), weights_only=False).to(device).eval()
    models = [model_context, model_body, emotic_model]

    # Initialize data storage
    frame_data = {
        'frame_numbers': [],
        'categories': {cat: [] for cat in ind2cat.values()},
        'person_detected': [],
        'skip_frames': skip_frames
    }

    # Open video stream
    video_stream = cv2.VideoCapture(video_file)
    print(f'Processing video with frame skipping (every {skip_frames+1} frames)')
    
    frame_count = 0
    processed_count = 0

    while True:
        grabbed, frame = video_stream.read()
        if not grabbed:
            break

        frame_count += 1

        # Skip frames according to parameter
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 1:
            continue

        processed_count += 1
        frame_data['frame_numbers'].append(frame_count)
        image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_detected = False

        try: 
            bbox_yolo = get_bbox(yolo, device, image_context)
            if len(bbox_yolo) > 0:
                person_detected = True
                for pred_bbox in bbox_yolo:
                    # Get emotion predictions (ignore VAD results with _)
                    pred_cat, _ = infer(context_norm, body_norm, ind2cat,
                                      device, thresholds, models,
                                      image_context=image_context,
                                      bbox=pred_bbox,
                                      to_print=False)
                    
                    # Track detected categories
                    for cat in ind2cat.values():
                        frame_data['categories'][cat].append(1 if cat in pred_cat else 0)

            frame_data['person_detected'].append(person_detected)

        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            frame_data['person_detected'].append(False)
            for cat in ind2cat.values():
                frame_data['categories'][cat].append(0)

    # Cleanup
    video_stream.release()
    
    # Save results
    save_video_results(frame_data, result_path, ind2cat)

    print(f'Processed {processed_count} frames (skipped every {skip_frames} frames)')

def save_video_results(frame_data, result_path, ind2cat):
    
    """Save results with proper validation and plotting."""
    # Create directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    # Calculate category percentages
    total_frames = len(frame_data['frame_numbers'])
    category_percent = {}

    for cat in ind2cat.values():
        if total_frames > 0:
            detection_count = sum(frame_data['categories'][cat])
            category_percent[cat] = (detection_count / total_frames) * 100
        else:
            category_percent[cat] = 0.0

    # Add to frame_data
    
    frame_data['category_percent'] = category_percent

    # Save raw data (JSON)
    with open(os.path.join(result_path, 'video_results.json'), 'w') as f:
        json.dump({'category_percent': category_percent}, f, indent=2)
