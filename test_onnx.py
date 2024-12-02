from onnxruntime import InferenceSession
import numpy as np
import cv2  
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.ops as ops  
import os

def execute_model(path, inputs):
    session = InferenceSession(path)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: inputs})
    
    # Print output names and shapes for verification
    for name, output in zip(output_names, outputs):
        print(f"Output '{name}': shape {output.shape}")
    
    return outputs

def decode_boxes(raw_boxes, anchors, x_scale, y_scale, w_scale, h_scale):
    boxes = torch.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(6):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset    ] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes

def tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors, scales, 
                          min_score_thresh=0.75, nms_threshold=0.3):
    x_scale, y_scale, w_scale, h_scale = scales
    detection_boxes = decode_boxes(raw_box_tensor, anchors, x_scale, y_scale, w_scale, h_scale)

    thresh = 100.0
    raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
    detection_scores = torch.sigmoid(raw_score_tensor).squeeze(dim=-1)

    mask = detection_scores >= min_score_thresh

    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]]

        if boxes.shape[0] == 0:
            output_detections.append(torch.zeros((0, 17)))
            continue

        # Apply Non-Maximum Suppression (NMS)
        keep_indices = ops.nms(boxes[:, :4], scores, nms_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices].unsqueeze(dim=-1)

        # Concatenate boxes with scores
        detections = torch.cat((boxes, scores), dim=-1)  # Shape: [num_detections, 17]

        output_detections.append(detections)

    return output_detections

def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print(f"Found {detections.shape[0]} face(s)")

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor="r", facecolor="none", 
                                 alpha=0.8)
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=3, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=0.8)
                ax.add_patch(circle)
        
    plt.show()

if __name__ == "__main__":
    path = "blazeface.onnx"

    # Path to the input image
    input_img_folder = "blaze_test_images"
    
    # Load and preprocess the input image
    for img_name in os.listdir(input_img_folder):
        input_img_path = os.path.join(input_img_folder, img_name)
        img_ = cv2.imread(input_img_path)
        if img_ is None:
            raise FileNotFoundError(f"Image not found at path: {input_img_path}")
        
        img = cv2.resize(img_, (128, 128))
        img = img.transpose(2, 0, 1)  # Change from HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        inputs = img.astype(np.float32)
        inputs = (inputs - 127.5) / 128.0  #! Model needs range [-1, 1]

        outputs = execute_model(path, inputs)

        if len(outputs) < 2:
            raise ValueError("The ONNX model does not have the expected number of outputs.")

        # Convert outputs to tensors
        raw_box_tensor = torch.tensor(outputs[0])
        raw_score_tensor = torch.tensor(outputs[1])

        # Load anchors
        anchors_path = "anchors.npy"
        anchors = np.load(anchors_path)
        anchors = torch.tensor(anchors, dtype=torch.float32)

        # Define scale parameters (adjust based on your model's configuration)
        x_scale = 128.0
        y_scale = 128.0
        w_scale = 128.0
        h_scale = 128.0

        scales = (x_scale, y_scale, w_scale, h_scale)
        detections = tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors, scales,
                                        min_score_thresh=0.75, nms_threshold=0.3)

        if detections and detections[0].shape[0] > 0:
            plot_detections(img_, detections[0].numpy())  # Assuming a single image
        else:
            print("No detections found.")
