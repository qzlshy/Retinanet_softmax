import torch
import numpy as np
import skimage.io as io
from torchvision import datasets, models, transforms
from retinanet import Retinanet
from anchors import anchor_targets_bbox
from losses import Focalloss, Label_Softmax_loss,Smooth_l1_loss
from filter_detections import bbox_transform_inv, clipBoxes, filter_detections
import os
import json
import loaddata_json

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


root_dir='/state/partition1/long/3-2_test'
img_files=os.listdir(root_dir)
img_path=[]
for a in img_files:
    img_path.append(os.path.join(root_dir,a))


model=Retinanet(2)
model.load_state_dict(torch.load('model_cell/retina_model.pth'))
model = model.to(device)

model.eval()
dic_rev={0:'cell0',1:'cell1'}

for img_file in img_path:
    img=io.imread(img_file)
    img=data_transforms(img)
    d1,d2,d3=img.shape
    img=img.reshape([1,d1,d2,d3])
    img = img.to(device)
    pre_r, pre_f, pre_c, all_anchors=model(img)
    all_anchors=torch.from_numpy(all_anchors).float()
    all_anchors=all_anchors.to(device)
    pred_boxes=bbox_transform_inv(all_anchors,pre_r)
    clip_boxes=clipBoxes(pred_boxes,img)
    p=filter_detections(clip_boxes,pre_f, pre_c,score_threshold = 0.2, nms_threshold = 0.2,max_detections = 30)
    box_tmp=[]
    box_pre=p[0].data.cpu().numpy()
    label_pre=p[2].data.cpu().numpy()
    for i in range(len(label_pre)):
        t={"label":dic_rev[label_pre[i]],"xmin":int(box_pre[i][0]),"xmax":int(box_pre[i][2]),"ymin":int(box_pre[i][1]),"ymax":int(box_pre[i][3])}
        box_tmp.append(t)
    out_file_name=img_file.split('/')[-1].split('.')[0]+'_pre.json'
    objs={"imgName":img_file.split('/')[-1],"objs":box_tmp}
    with open(out_file_name,"w") as f_obj:
        json.dump(objs,f_obj)


