import onnxruntime as ort
import numpy as np
import urllib.request
import os

# ⚠️ 已被 probe_coord_space.py 取代（本脚本的 cxcywh 解码假设是错的，仅留作过程记录）。
# 已知错误：col2-5 实为 x1y1x2y2（非 cxcywh）；outs[1] 是 count（非 class_id）。
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.onnx')

# Download official demo image
demo_url = 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg'
local_demo = os.path.expanduser('~/.cache/mica-ai-layout/layout_demo.jpg')
os.makedirs(os.path.dirname(local_demo), exist_ok=True)
if not os.path.exists(local_demo):
    urllib.request.urlretrieve(demo_url, local_demo)
print(f'demo image: {local_demo} {os.path.getsize(local_demo)} bytes')

# Load and preprocess with PIL
from PIL import Image
img = Image.open(local_demo).convert('RGB')
W0, H0 = img.size
print(f'orig size: {W0} x {H0}')

# Letterbox to 800x800, BGR→RGB, CHW, scale to [0,1]
target = 800
r = min(target / W0, target / H0)
new_w = int(round(W0 * r))
new_h = int(round(H0 * r))
pad_w = target - new_w
pad_h = target - new_h
img_resized = img.resize((new_w, new_h), Image.BILINEAR)
padded = Image.new('RGB', (target, target), (114, 114, 114))
padded.paste(img_resized, (0, 0))

arr = np.array(padded).astype(np.float32) / 255.0  # HWC RGB
mean = np.array([0.8286, 0.8281, 0.8282], dtype=np.float32)
std = np.array([0.1889, 0.1889, 0.1889], dtype=np.float32)
chw = (arr - mean) / std
chw = chw.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # [1,3,800,800]

# PaddleX V3: im_shape = [1, 2]
im_shape = np.array([[float(H0), float(W0)]], dtype=np.float32)
# scale_factor uses orig / padded scale
scale_factor = np.array([[float(H0) / float(new_h), float(W0) / float(new_w)]], dtype=np.float32)

# Load model
m = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
feeds = {'image': chw, 'im_shape': im_shape, 'scale_factor': scale_factor}
outs = m.run(None, feeds)

boxes_out = outs[0]
class_id_out = outs[1]
order_out = outs[2]

print(f'\nboxes shape: {boxes_out.shape}')
print(f'class_id shape: {class_id_out.shape}')
print(f'order shape: {order_out.shape}')

print('\n=== boxes sample ===')
for i in range(min(5, len(boxes_out))):
    print(f'  [{i}] {boxes_out[i].tolist()}')

print('\n=== class_id ===')
print(f'  unique values: {np.unique(class_id_out).tolist()}')

# Decode boxes — guess col0 = class, col1 = score, col2-5 = cxcywh normalized
print('\n=== decode guess: col0=cls, col1=score, col2-5=cxcywh (norm to 800) ===')
boxes = boxes_out
for i in range(min(20, len(boxes))):
    cls = int(round(boxes[i][0]))
    score = boxes[i][1]
    cx, cy, w, h = boxes[i][2:6]
    if score < 0.3: continue
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    print(f'  cls={cls} score={score:.3f} bbox(800)=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')

# Apply NMS at low threshold to see how many real detections
print('\n=== high-score detections (score > 0.4) ===')
for i in range(len(boxes)):
    cls = int(round(boxes[i][0]))
    score = boxes[i][1]
    if score < 0.4: continue
    cx, cy, w, h = boxes[i][2:6]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    print(f'  cls={cls} score={score:.3f} bbox(800)=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')

print('\n=== order_index (N, 200, 200) shape - sample first 5x5 for first detection ===')
print(order_out[0, :5, :5])