import onnxruntime as ort
import numpy as np
import os

# 模型 I/O 元信息探针：换模型 / 怀疑导出约定变化时先跑这个。
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.onnx')

m = ort.InferenceSession(MODEL,
                        providers=['CPUExecutionProvider'])

print('=== Inputs ===')
for inp in m.get_inputs():
    print(f'  {inp.name}: shape={inp.shape} dtype={inp.type}')

print('=== Outputs ===')
for out in m.get_outputs():
    print(f'  {out.name}: shape={out.shape} dtype={out.type}')

# Run a dummy inference to inspect real shapes
dummy_image = np.zeros((1, 3, 800, 800), dtype=np.float32)
# PaddleX V3 actual: im_shape is [1, 2] = [orig_h, orig_w] (NO 3rd dim)
dummy_im_shape = np.array([[800, 800]], dtype=np.float32)
dummy_scale = np.array([[1.0, 1.0]], dtype=np.float32)

feeds = {'image': dummy_image, 'im_shape': dummy_im_shape, 'scale_factor': dummy_scale}
try:
    outs = m.run(None, feeds)
    print('=== Output shapes after dummy run ===')
    for name, arr in zip([o.name for o in m.get_outputs()], outs):
        print(f'  {name}: shape={arr.shape} dtype={arr.dtype}')
        if arr.size > 0 and arr.ndim >= 2:
            print(f'    sample row[0] = {arr.flatten()[:6].tolist()}')
except Exception as e:
    print(f'run failed: {e}')