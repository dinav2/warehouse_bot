import zmq
import numpy as np
import cv2
import os

# Conectar al servidor ZMQ
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

print("Esperando imagenes del servidor ZMQ")
print("Presiona [Espacio] para capturar una imagen, [q] para salir.\n")

capturas = []
i = 0

while True:
    try:
        socket.send(b"get")  # Envia solicitud de imagen

        jpg_bytes = socket.recv()
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)

        cv2.imshow("Vista para calibracion", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            print("Captura almacenada ({})".format(i + 1))
            capturas.append(frame.copy())
            i += 1
        elif key == ord("q"):
            break
    except Exception as e:
        print("Error al recibir imagen:", e)
        break

cv2.destroyAllWindows()

if len(capturas) == 0:
    print("No se capturaron imagenes.")
else:
    print("Se capturaron {} imagenes para calibracion.".format(len(capturas)))

    # Guardar capturas opcionalmente
    os.makedirs("calib", exist_ok=True)
    for j, img in enumerate(capturas):
        cv2.imwrite("calib/img_{}.jpg".format(j), img)
    print("Imagenes guardadas en la carpeta 'calib/'.")

