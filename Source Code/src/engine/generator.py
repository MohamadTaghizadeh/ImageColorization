import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from loguru import logger

# Emotion categories - exactly as in your original code
CATEGORIES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence',
    'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment',
    'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain',
    'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
    'Sympathy', 'Yearning'
]

# Create mappings exactly as in your original main.py
cat2ind = {}
ind2cat = {}
for idx, emotion in enumerate(CATEGORIES):
    cat2ind[emotion] = idx
    ind2cat[idx] = emotion

vad = ['Valence', 'Arousal', 'Dominance']
ind2vad = {}
for idx, continuous in enumerate(vad):
    ind2vad[idx] = continuous

# Emotic class - exactly from your emotic.py
class Emotic(nn.Module):
    """ Emotic Model"""
    def __init__(self, num_context_features, num_body_features):
        super(Emotic, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((self.num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out

# YOLO utility functions - exactly from your yolo_utils.py
def to_cpu(tensor):
    return tensor.detach().cpu()

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def rescale_boxes(boxes, current_dim, original_shape):
    orig_h, orig_w = original_shape
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        if not image_pred.size(0):
            continue
            
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
            
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

# All YOLO classes - exactly from your yolo_utils.py
def parse_model_config(path):
    with open(path, 'r') as file:
        lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs

def create_modules(module_defs):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
            
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        return output, 0

class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
            
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)

        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

def prepare_yolo(model_dir):
    cfg_file = os.path.join(model_dir, 'yolov3.cfg')
    weight_file = os.path.join(model_dir, 'yolov3.weights')
    
    if not os.path.exists(cfg_file):
        logger.error(f"YOLO config file not found: {cfg_file}")
        raise FileNotFoundError(f"YOLO config file not found: {cfg_file}")
    
    if not os.path.exists(weight_file):
        logger.error(f"YOLO weights file not found: {weight_file}")
        raise FileNotFoundError(f"YOLO weights file not found: {weight_file}")
    
    yolo_model = Darknet(cfg_file, 416)
    yolo_model.load_darknet_weights(weight_file)
    logger.info('Prepared YOLO model')
    return yolo_model

# Functions from your inference.py
def process_images(context_norm, body_norm, image_context=None, image_body=None, bbox=None):
    """Process images exactly as in your inference.py"""
    if image_context is None:
        raise ValueError('image_context cannot be none')
    if image_body is None and bbox is None: 
        raise ValueError('both body image and bounding box cannot be none')

    if bbox is not None:
        image_body = image_context[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
    
    image_context = cv2.resize(image_context, (224, 224))
    image_body = cv2.resize(image_body, (128, 128))
    
    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    context_norm_transform = transforms.Normalize(context_norm[0], context_norm[1])  
    body_norm_transform = transforms.Normalize(body_norm[0], body_norm[1])

    image_context = context_norm_transform(test_transform(image_context)).unsqueeze(0)
    image_body = body_norm_transform(test_transform(image_body)).unsqueeze(0)

    return image_context, image_body

def infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=None, bbox=None, to_print=False):
    """Infer function exactly as in your inference.py"""
    image_context_tensor, image_body_tensor = process_images(context_norm, body_norm, image_context=image_context, bbox=bbox)

    model_context, model_body, emotic_model = models
    
    with torch.no_grad():
        image_context_tensor = image_context_tensor.to(device)
        image_body_tensor = image_body_tensor.to(device)
        
        pred_context = model_context(image_context_tensor)
        pred_body = model_body(image_body_tensor)
        pred_cat, pred_cont = emotic_model(pred_context, pred_body)
        pred_cat = pred_cat.squeeze(0)
        pred_cont = pred_cont.squeeze(0).to("cpu").data.numpy()

        bool_cat_pred = torch.gt(pred_cat, thresholds)
    
    cat_emotions = list()
    for i in range(len(bool_cat_pred)):
        if bool_cat_pred[i] == True:
            cat_emotions.append(ind2cat[i])

    if to_print == True:
        logger.info(f'Categorical Emotions: {cat_emotions}')
        logger.info(f'VAD values: {pred_cont}')
    
    return cat_emotions, 10*pred_cont

def get_bbox(yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):
    """Get bbox function exactly as in your yolo_inference.py"""
    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = yolo_model(image_yolo)
        nms_det = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
        det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))
    
    bboxes = []
    for x1, y1, x2, y2, _, _, cls_pred in det:
        if cls_pred == 0:  # checking if predicted_class = persons
            x1 = int(min(image_context.shape[1], max(0, x1)))
            x2 = int(min(image_context.shape[1], max(x1, x2)))
            y1 = int(min(image_context.shape[0], max(15, y1)))
            y2 = int(min(image_context.shape[0], max(y1, y2)))
            bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes)

# Main EmotionDetector class - exactly following your yolo_video logic
class EmotionDetector:
    def __init__(self, gpu=0):
        # Exactly as in your yolo_video function
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Define paths
        if os.environ.get("MODE", "dev") == "prod":
            model_dir = "/approot/models"
        else:
            model_dir = os.path.normpath("../../../Models")
        
        logger.info(f"Model directory: {model_dir}")
        
        # Load YOLO model first
        self.yolo = prepare_yolo(model_dir).to(self.device).eval()
        
        # Load thresholds
        self.thresholds = torch.FloatTensor(np.load(
            os.path.join(model_dir, 'val_thresholds.npy')
        )).to(self.device)
        
        # Load models with proper error handling for class loading issues
        try:
            # Try to load models normally first
            self.model_context = torch.load(
                os.path.join(model_dir, 'model_context1.pth'), 
                weights_only=False,
                map_location=self.device
            ).eval()
            
            self.model_body = torch.load(
                os.path.join(model_dir, 'model_body1.pth'), 
                weights_only=False,
                map_location=self.device
            ).eval()
            
            self.emotic_model = torch.load(
                os.path.join(model_dir, 'model_emotic1.pth'), 
                weights_only=False,
                map_location=self.device
            ).eval()
            
            logger.info("Models loaded successfully using direct loading")
            
        except Exception as e:
            logger.warning(f"Direct model loading failed: {e}")
            logger.info("Attempting to load using state_dict method...")
            
            try:
                # Load state dictionaries and create new model instances
                # For emotic model - we know its structure
                self.emotic_model = Emotic(num_context_features=256, num_body_features=256)
                emotic_checkpoint = torch.load(
                    os.path.join(model_dir, 'model_emotic1.pth'),
                    map_location=self.device,
                    weights_only=False
                )
                
                # Handle different checkpoint formats
                if hasattr(emotic_checkpoint, 'state_dict'):
                    self.emotic_model.load_state_dict(emotic_checkpoint.state_dict())
                elif isinstance(emotic_checkpoint, dict) and 'state_dict' in emotic_checkpoint:
                    self.emotic_model.load_state_dict(emotic_checkpoint['state_dict'])
                elif isinstance(emotic_checkpoint, dict):
                    self.emotic_model.load_state_dict(emotic_checkpoint)
                else:
                    # If it's already a model, extract state dict
                    self.emotic_model.load_state_dict(emotic_checkpoint.state_dict())
                
                self.emotic_model = self.emotic_model.to(self.device).eval()
                
                # For context and body models, try to load them as-is since they're likely standard models
                self.model_context = torch.load(
                    os.path.join(model_dir, 'model_context1.pth'),
                    map_location=self.device,
                    weights_only=False
                ).eval()
                
                self.model_body = torch.load(
                    os.path.join(model_dir, 'model_body1.pth'),
                    map_location=self.device, 
                    weights_only=False
                ).eval()
                
                logger.info("Models loaded successfully using state_dict method")
                
            except Exception as e2:
                logger.error(f"Both loading methods failed: {e}, {e2}")
                raise Exception(f"Cannot load models: {e2}")
        
        self.models = [self.model_context, self.model_body, self.emotic_model]
        
        # Normalization exactly as in your original code
        self.context_mean = [0.4690646, 0.4407227, 0.40508908]
        self.context_std = [0.2514227, 0.24312855, 0.24266963]
        self.body_mean = [0.43832874, 0.3964344, 0.3706214]
        self.body_std = [0.24784276, 0.23621225, 0.2323653]
        self.context_norm = [self.context_mean, self.context_std]
        self.body_norm = [self.body_mean, self.body_std]
        
        logger.info("All models loaded successfully")

    def process_video(self, video_path, skip_frames=9):
        """Process video exactly as in your yolo_video function"""
        logger.info(f"Processing video: {video_path}")
        
        # Initialize data storage exactly as in your yolo_video
        frame_data = {
            'frame_numbers': [],
            'categories': {cat: [] for cat in ind2cat.values()},
            'person_detected': [],
            'skip_frames': skip_frames
        }

        video_stream = cv2.VideoCapture(video_path)
        if not video_stream.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return {"error": f"Cannot open video: {video_path}"}
        
        logger.info(f'Processing video with frame skipping (every {skip_frames+1} frames)')
        
        frame_count = 0
        processed_count = 0

        try:
            while True:
                grabbed, frame = video_stream.read()
                if not grabbed:
                    break

                frame_count += 1

                # Skip frames exactly as in your yolo_video
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 1:
                    continue

                processed_count += 1
                frame_data['frame_numbers'].append(frame_count)
                image_context = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                person_detected = False

                try: 
                    bbox_yolo = get_bbox(self.yolo, self.device, image_context)
                    if len(bbox_yolo) > 0:
                        person_detected = True
                        for pred_bbox in bbox_yolo:
                            # Use the exact infer function from your working code
                            pred_cat, _ = infer(
                                self.context_norm, 
                                self.body_norm, 
                                ind2cat, 
                                ind2vad,
                                self.device, 
                                self.thresholds, 
                                self.models,
                                image_context=image_context,
                                bbox=pred_bbox,
                                to_print=False
                            )
                            
                            # Track detected categories exactly as in your working code
                            for cat in ind2cat.values():
                                frame_data['categories'][cat].append(1 if cat in pred_cat else 0)

                    # Handle frames without detections
                    frame_data['person_detected'].append(person_detected)
                    if not person_detected:
                        for cat in ind2cat.values():
                            frame_data['categories'][cat].append(0)
                            
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                    frame_data['person_detected'].append(False)
                    for cat in ind2cat.values():
                        frame_data['categories'][cat].append(0)

        finally:
            video_stream.release()

        # Calculate percentages exactly as in your working code
        total_frames = len(frame_data['frame_numbers'])
        category_percent = {}
        
        for cat in ind2cat.values():
            if total_frames > 0:
                detection_count = sum(frame_data['categories'][cat])
                category_percent[cat] = (detection_count / total_frames) * 100
            else:
                category_percent[cat] = 0.0
        
        # Final results exactly matching your working output format
        results = {
            "total_frames": frame_count,
            "frames_processed": processed_count,
            "emotion_percentages": category_percent
        }
        
        logger.info(f'Processed {processed_count} frames, total frames: {frame_count}')
        logger.info(f'Emotion results: {results["emotion_percentages"]}')
        
        return results