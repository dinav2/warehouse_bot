# signals.py
import cv2
import numpy as np
import math
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = engine.get_binding_shape(i)
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        io = {"host": host_mem, "device": device_mem, "shape": shape}
        (inputs if engine.binding_is_input(i) else outputs).append(io)

    return context, inputs, outputs, bindings, stream

def preprocess(img):
    resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = img_norm.transpose((2, 0, 1))
    return np.expand_dims(img_chw, axis=0)

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def nms(boxes, iou_thres):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    selected = []
    while boxes:
        chosen = boxes.pop(0)
        selected.append(chosen)
        boxes = [b for b in boxes if iou(chosen, b) < iou_thres]
    return selected

def postprocess(output, img):
    pred = output.reshape(1, 10, -1).transpose(0, 2, 1)[0]
    boxes, conf, probs = pred[:, :4], pred[:, 4:5], pred[:, 5:]
    scores = conf * probs
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    results = []
    for i in range(len(boxes)):
        if confidences[i] < 0.0001:
            continue
        cx, cy, w, h = boxes[i]
        x1 = (cx - w / 2) * img.shape[1] / 640
        y1 = (cy - h / 2) * img.shape[0] / 640
        x2 = (cx + w / 2) * img.shape[1] / 640
        y2 = (cy + h / 2) * img.shape[0] / 640
        results.append([x1, y1, x2, y2, confidences[i], class_ids[i]])

    return nms(results, 0.5)
