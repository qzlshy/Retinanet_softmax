import numpy as np
import torch
import torchvision



def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    x1 = boxes[:, 0] + (deltas[:, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, 1] + (deltas[:, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, 2] + (deltas[:, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, 3] + (deltas[:, 3] * std[3] + mean[3]) * height

    pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    return pred_boxes


def clipBoxes(boxes, image):
    _, _, height, width=image.shape
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    x1=torch.clamp(x1, min=0, max=width)
    y1=torch.clamp(y1, min=0, max=height)
    x2=torch.clamp(x2, min=0, max=width)
    y2=torch.clamp(y2, min=0, max=height)
    clip_boxes=torch.stack([x1, y1, x2, y2], dim=1)
    return clip_boxes


def filter_detections(boxes, frount, classification, nms = True, score_threshold = 0.05, max_detections = 50, nms_threshold = 0.5):

    def _filter_detections(scores, labels):
        indices = torch.nonzero(scores>score_threshold)[:,0]
        if nms:
            filtered_boxes=boxes[indices]
            filtered_scores=scores[indices]
            nms_indices = torchvision.ops.nms(filtered_boxes, filtered_scores, nms_threshold)
            indices = indices[nms_indices][:max_detections]

        labels = labels[indices]
        return indices, labels

    scores = frount.view(-1)
    labels = torch.argmax(classification, dim=1)
    indices, labels = _filter_detections(scores, labels)

     
    scores = scores[indices]
    scores, top_indices = torch.topk(scores, min(max_detections,scores.shape[0]))

    indices=indices[top_indices]
    boxes=boxes[indices]
    labels=labels[top_indices]

    return [boxes, scores, labels]



