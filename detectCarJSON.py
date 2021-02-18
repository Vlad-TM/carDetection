import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from time import sleep
import json
import re


def detect(img):
    outputVector = []
    img_size, out, source, weights = opt.img_size, opt.output, opt.source, opt.weights

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    t = time.time()

    # Get detections

    im0s = img

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            # Write results
            for *xyxy, conf, _, cls in det:
                if cls == 2 or cls == 7:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color = [0,255,0] )#color = colors[int(cls)])
                    x1 = xyxy[0].cpu().numpy()
                    y1 = xyxy[1].cpu().numpy()
                    x2 = xyxy[2].cpu().numpy()
                    y2 = xyxy[3].cpu().numpy()
                    maxx = x1 if x1 > x2 else x2
                    maxy = y1 if y1 > y2 else y2
                    minx = x1 if x1 < x2 else x2
                    miny = y1 if y1 < y2 else y2

                    outputVector.append([int(minx),int(miny),int(maxx),int(maxy)])

        print('%sDone. (%.3fs)' % (s, time.time() - t))
        cv2.imshow('im', im0)
        cv2.waitKey(1)
        return outputVector

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/yolov3-tiny.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='./data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='./weights/yolov3-tiny.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='./input', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():

        cap = cv2.VideoCapture('/home/vlade/Downloads/output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        out = cv2.VideoWriter('fresult.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         60, size)
        nFrame = 0
        dictForJSON = {}
        while True:
            nFrame += 1
            try:
                _, frame = cap.read()
                bboxs = detect(cv2.resize(frame, None, fx=0.25, fy=0.25))
            except:
                print('end video')
                break

            currList = []
            for bbox in bboxs:
                currDict = {}
                bbox = [i * 4 for i in bbox]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 0), thickness=5)
                currDict['x_left'], currDict['y_top'], currDict['x_right'], currDict['y_bottom'] = bbox
                currList.append(currDict)
            out.write(frame)
            dictForJSON['frame_' + str(nFrame)] = currList

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        with open('car.json', 'w') as out_file:
            str = json.dumps(dictForJSON, indent=4)
            str = re.sub(r"(:\s\d+,)\s*", r'\1', str)
            str = re.sub(r"{\s*(.*\d+)\s*}", r"{\1}", str)
            out_file.write(str)
        cap.release()
        out.release()
        cv2.destroyAllWindows()