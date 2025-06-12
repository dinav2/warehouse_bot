import zmq
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2.aruco as aruco
import math
import json

# Diccionario de poses de ArUcos (en metros)
poses = {
    1: {"x": 2.45, "y": 0.30, "theta": math.pi},
    2: {"x": 1.52, "y": 2.45, "theta": -math.pi / 2},
    3: {"x": 0.00, "y": 2.15, "theta": 0.0},
    4: {"x": 1.04, "y": 0.65, "theta": math.pi / 2},
    5: {"x": 0.00, "y": 1.53, "theta": 0.0},
    6: {"x": 1.04, "y": 2.45, "theta": -math.pi / 2},
    7: {"x": 0.00, "y": 0.90, "theta": 0.0},
    8: {"x": 2.02, "y": 0.00, "theta": math.pi / 2},
    9: {"x": 0.00, "y": 0.33, "theta": 0.0},
    10: {"x": 1.53, "y": 0.94, "theta": math.pi},
    11: {"x": 1.53, "y": 1.54, "theta": math.pi},
    12: {"x": 2.45, "y": 0.74, "theta": math.pi},
    13: {"x": 2.45, "y": 1.28, "theta": math.pi},
    14: {"x": 2.03, "y": 2.45, "theta": -math.pi / 2},
    15: {"x": 0.33, "y": 1.24, "theta": math.pi / 2},
    16: {"x": 0.50, "y": 0.65, "theta": math.pi / 2},
    17: {"x": 0.50, "y": 2.45, "theta": -math.pi / 2},
    18: {"x": 2.45, "y": 2.16, "theta": math.pi}
}

ENGINE_PATH = "/root/modelos/signals.engine"
CONF_THRESHOLD = 0.0001
IOU_THRESHOLD = 0.5
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
MAX_DETECTIONS = 100

data = np.load("/root/tsuru_ws/calibracion.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

marker_length = 0.045
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

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

def preprocess(img):
    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = img_norm.transpose((2, 0, 1))
    return np.expand_dims(img_chw, axis=0)

def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
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
    pred = output.reshape(1, 10, -1).transpose(0, 2, 1)[0]
    boxes, obj_conf, class_probs = pred[:, :4], pred[:, 4:5], pred[:, 5:]
    scores = obj_conf * class_probs
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    results = []
    for i in range(len(boxes)):
        if confidences[i] < CONF_THRESHOLD:
            continue
        cx, cy, w, h = boxes[i]
        x1 = (cx - w / 2) * img.shape[1] / INPUT_WIDTH
        y1 = (cy - h / 2) * img.shape[0] / INPUT_HEIGHT
        x2 = (cx + w / 2) * img.shape[1] / INPUT_WIDTH
        y2 = (cy + h / 2) * img.shape[0] / INPUT_HEIGHT
        results.append([x1, y1, x2, y2, confidences[i], class_ids[i]])
    final_boxes = nms(results, IOU_THRESHOLD)
    return final_boxes

context = zmq.Context()

rep_socket = context.socket(zmq.REP)
rep_socket.bind("tcp://*:5555")

pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:6001")

print("Servidor ZMQ REP en puerto 5555 y PUB en puerto 6001")

context_trt, inputs, outputs, bindings, stream = load_engine(ENGINE_PATH)

while True:
    jpg_bytes = rep_socket.recv()
    frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    input_tensor = preprocess(frame)
    np.copyto(inputs[0]["host"], input_tensor.ravel())

    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context_trt.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()

    output = outputs[0]["host"]
    annotated = frame.copy()

    final_boxes = postprocess(output, annotated)

    clase_detectada = "None"
    confidences = []
    class_ids = []

    if final_boxes:
        confidences = [box[4] for box in final_boxes]
        class_ids = [box[5] for box in final_boxes]
        if np.max(confidences) > CONF_THRESHOLD:
            clase_detectada = str(class_ids[np.argmax(confidences)])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        ids_flat = ids.flatten()
        valid_ids = [int(i) for i in ids_flat if i in poses]
        if valid_ids:
            min_id = int(min(valid_ids))
            i = np.where(ids_flat == min_id)[0][0]
            _, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            tvec = tvecs[i][0]
            pose = poses[min_id]
            theta = pose["theta"]
            if theta == 0.0:
                x_robot = float(pose["x"] + tvec[2])
                y_robot = float(pose["y"] - tvec[0])
            elif theta == math.pi:
                x_robot = float(pose["x"] - tvec[2])
                y_robot = float(pose["y"] + tvec[0])
            elif theta == math.pi / 2:
                x_robot = float(pose["x"] + tvec[0])
                y_robot = float(pose["y"] + tvec[2])
            elif theta == -math.pi / 2:
                x_robot = float(pose["x"] - tvec[0])
                y_robot = float(pose["y"] - tvec[2])
            else:
                x_robot, y_robot = 0.0, 0.0

            x_robot -= 0.18
            y_robot -= 0.285
            yaw_robot = float((theta + math.pi) % (2 * math.pi))

            info = {
                "aruco_id": min_id,
                "x": x_robot,
                "y": y_robot,
                "yaw": yaw_robot,
                "clase": clase_detectada
            }

            mensaje = json.dumps(info)
            pub_socket.send_string(mensaje)
            print("[ZMQ] Enviado:", mensaje)
        else:
            pub_socket.send_string(json.dumps({
                "aruco_id": None,
                "x": None,
                "y": None,
                "yaw": None,
                "clase": clase_detectada
            }))
    else:
        pub_socket.send_string(json.dumps({
            "aruco_id": None,
            "x": None,
            "y": None,
            "yaw": None,
            "clase": clase_detectada
        }))

    _, encoded = cv2.imencode(".jpg", annotated)
    rep_socket.send(encoded.tobytes())


cuda.Context.get_current().pop()
