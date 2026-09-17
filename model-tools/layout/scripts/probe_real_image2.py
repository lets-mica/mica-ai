import onnxruntime as ort
import numpy as np
import urllib.request
import os

# ⚠️ 已被 probe_coord_space.py 取代，仅留作过程记录（原始 300 行 dump）。
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.onnx')

# Download official demo image
demo_url = 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg'
local_demo = os.path.expanduser('~/.cache/mica-ai-layout/layout_demo.jpg')
os.makedirs(os.path.dirname(local_demo), exist_ok=True)
if not os.path.exists(local_demo):
    urllib.request.urlretrieve(demo_url, local_demo)

from PIL import Image
img = Image.open(local_demo).convert('RGB')
W0, H0 = img.size
target = 800
r = min(target / W0, target / H0)
new_w = int(round(W0 * r))
new_h = int(round(H0 * r))
pad_w = target - new_w
pad_h = target - new_h
print(f'orig {W0}x{H0} → letterbox {new_w}x{new_h} (r={r:.4f}), pad=({pad_w},{pad_h})')

img_resized = img.resize((new_w, new_h), Image.BILINEAR)
padded = Image.new('RGB', (target, target), (114, 114, 114))
padded.paste(img_resized, (0, 0))

arr = np.array(padded).astype(np.float32) / 255.0
mean = np.array([0.8286, 0.8281, 0.8282], dtype=np.float32)
std = np.array([0.1889, 0.1889, 0.1889], dtype=np.float32)
chw = (arr - mean) / std
chw = chw.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

# PaddleX V3 im_shape actual: [1, 2] = [orig_h, orig_w]
im_shape = np.array([[float(H0), float(W0)]], dtype=np.float32)
scale_factor = np.array([[float(H0) / float(new_h), float(W0) / float(new_w)]], dtype=np.float32)

m = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
feeds = {'image': chw, 'im_shape': im_shape, 'scale_factor': scale_factor}
outs = m.run(None, feeds)
boxes_out, count_out, order_out = outs

print(f'\ncount_out = {count_out}')  # should be # of valid detections

# Look at first 10 box rows in detail
print('\n=== first 10 rows of boxes_out ([300,7]) ===')
for i in range(10):
    row = boxes_out[i].tolist()
    print(f'  [{i}] col0_cls={row[0]:.0f} col1_score={row[1]:.4f} '
          f'col2_x1={row[2]:.1f} col3_y1={row[3]:.1f} col4_x2={row[4]:.1f} col5_y2={row[5]:.1f} '
          f'col6_?={row[6]:.4f}')

# Determine if col6 is a second score or a duplicate cls
print('\n=== col6 vs col1 correlation ===')
col1 = boxes_out[:50, 1]
col6 = boxes_out[:50, 6]
print('col1[:10] =', col1[:10])
print('col6[:10] =', col6[:10])

# After reverse-letterbox: convert (x1, y1) in 800-space to orig coords
# x1_orig = (x1_padded - 0) / r  (no pad in this layout)
print('\n=== top 10 detections after reverse-letterbox to orig (1654x2339) ===')
print(f'W0={W0} H0={H0} r={r:.4f}')
for i in range(min(20, len(boxes_out))):
    cls = int(round(boxes_out[i][0]))
    score = boxes_out[i][1]
    if score < 0.4: continue
    x1 = boxes_out[i][2] / r
    y1 = boxes_out[i][3] / r
    x2 = boxes_out[i][4] / r
    y2 = boxes_out[i][5] / r
    print(f'  cls={cls} score={score:.3f} orig_bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')

print(f'\norder_out shape: {order_out.shape} dtype: {order_out.dtype}')
print('order_out[0, :5, :5] =')
print(order_out[0, :5, :5])
print('order_out nonzero total:', np.count_nonzero(order_out))