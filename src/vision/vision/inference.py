import zmq
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# CONFIG
ENGINE_PATH = "/root/modelos/signals.engine"
CONF_THRESHOLD = 0.0001
IOU_THRESHOLD = 0.5
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
MAX_DETECTIONS = 100
NUM_CLASSES = 4

# LOAD ENGINE
def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        is_input = engine.binding_is_input(i)
        shape = engine.get_binding_shape(i)
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        io = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}
        (inputs if is_input else outputs).append(io)

    return context, inputs, outputs, bindings, stream

# PRE/POST PROCESSING
def preprocess(img):
    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = img_norm.transpose((2, 0, 1))
    return np.expand_dims(img_chw, axis=0)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

def nms(boxes, iou_thres):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    selected = []
    while boxes:
        chosen = boxes.pop(0)
        selected.append(chosen)
        boxes = [b for b in boxes if iou(chosen, b) < iou_thres]
    return selected

def postprocess(output, img):
    pred = output.reshape(1, 9, -1).transpose(0, 2, 1)[0]  # (N, 9)
    boxes = pred[:, :4]
    obj_conf = pred[:, 4:5]
    class_probs = pred[:, 5:]
    scores = obj_conf * class_probs

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    print("[DEBUG] Total predicciones crudas:", len(pred))
    print("[DEBUG] Maxima confianza:", np.max(confidences))

    results = []
    for i in range(len(boxes)):
        if confidences[i] < CONF_THRESHOLD:
            continue
        print("[DEBUG] Conf:", confidences[i], "Clase:", class_ids[i])
        cx, cy, w, h = boxes[i]
        x1 = (cx - w / 2) * img.shape[1] / INPUT_WIDTH
        y1 = (cy - h / 2) * img.shape[0] / INPUT_HEIGHT
        x2 = (cx + w / 2) * img.shape[1] / INPUT_WIDTH
        y2 = (cy + h / 2) * img.shape[0] / INPUT_HEIGHT
        results.append([x1, y1, x2, y2, confidences[i], class_ids[i]])

    final_boxes = nms(results, IOU_THRESHOLD)

    for box in final_boxes[:MAX_DETECTIONS]:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{int(cls)}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return img

# MAIN
context, inputs, outputs, bindings, stream = load_engine(ENGINE_PATH)

socket = zmq.Context().socket(zmq.REP)
socket.bind("tcp://*:5555")
print("Servidor de inferencia listo en puerto 5555")

while True:
    jpg_bytes = socket.recv()
    frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    input_tensor = preprocess(frame)
    np.copyto(inputs[0]["host"], input_tensor.ravel())

    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    output = outputs[0]["host"]
    annotated = postprocess(output, frame.copy())
    _, encoded = cv2.imencode(".jpg", annotated)
    socket.send(encoded.tobytes())
