import onnxruntime as ort
import numpy as np
import urllib.request, os
from PIL import Image

# ⚠️ 已被 probe_coord_space.py 取代，仅留作过程记录（order 矩阵早期探查）。
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.onnx')

# Reuse demo
local_demo = os.path.expanduser('~/.cache/mica-ai-layout/layout_demo.jpg')
img = Image.open(local_demo).convert('RGB')
W0, H0 = img.size
target = 800
r = min(target / W0, target / H0)
new_w = int(round(W0 * r))
new_h = int(round(H0 * r))
img_resized = img.resize((new_w, new_h), Image.BILINEAR)
padded = Image.new('RGB', (target, target), (114, 114, 114))
padded.paste(img_resized, (0, 0))

arr = np.array(padded).astype(np.float32) / 255.0
mean = np.array([0.8286, 0.8281, 0.8282], dtype=np.float32)
std = np.array([0.1889, 0.1889, 0.1889], dtype=np.float32)
chw = (arr - mean) / std
chw = chw.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

im_shape = np.array([[float(H0), float(W0)]], dtype=np.float32)
scale_factor = np.array([[float(H0) / float(new_h), float(W0) / float(new_w)]], dtype=np.float32)

m = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
feeds = {'image': chw, 'im_shape': im_shape, 'scale_factor': scale_factor}
boxes_out, count_out, order_out = m.run(None, feeds)

print(f'order_out shape={order_out.shape} dtype={order_out.dtype}')
print(f'order_out min={order_out.min()} max={order_out.max()} unique={np.unique(order_out)[:20]}')
print(f'order_out first row nonzero cols:')
for r in range(0, 15):
    nzs = np.nonzero(order_out[r])[0]
    if len(nzs) > 0:
        print(f'  row[{r}] nonzero at cols {nzs[:10].tolist()}, vals={order_out[r, nzs[:5]].tolist()}')

# Check if 300 is hard cap
print('\norder_out shape last 2 dims = 200x200 always?')
print('check: order_out[:200, 200, 200].shape =', order_out.shape)