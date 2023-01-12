import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import uuid
import re
import os
import requests
import pickle
import face_recognition
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *

from datetime import datetime
cameraUrl = os.environ['CAMERA_URL']
cameraType = os.environ['CAMERA_TYPE']

dataset_names = []
dataset = os.walk("dataset")
known_face_encodings = []
for data in dataset:
    dataset_names = data[2]
    for dataset_name in data[2]:
        loaded_dataset = pickle.loads(open("dataset/" + dataset_name, "rb").read())
        print(dataset_name, 'dataset_name')
        known_face_encodings.append(loaded_dataset)
print(dataset_names, 'dataset_names')


def detect_person_in_video(image):
    rframe = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    locations = face_recognition.face_locations(rframe)
    encodings = face_recognition.face_encodings(rframe, locations)
    datasets_names = []
    for face_encoding in encodings:
        matches = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"
        print(matches, 'matches')
        if len(matches):
            best_match_index = np.argmin(matches)
            if matches[best_match_index] and matches[best_match_index] < 0.6:
                name = dataset_names[best_match_index]
                datasets_names.append(name)

    return datasets_names

detected_dataset_names = ['Unknown'] * 1000000

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    source = cameraUrl
    frames = 0
    startTime = datetime.today()
    startTimeSeconds = startTime.timestamp()
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    people = 0
    ip = re.findall(r'(?:\d{1,3}\.)+(?:\d{1,3})', source)
    save_photo_dir = 'images/' + ip[0] + '/'
    if not os.path.exists(save_photo_dir):
        os.mkdir(save_photo_dir)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = 0
    ###################################
    for path, img, im0s, vid_cap in dataset:
        # frames += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0, 6))
                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))

                if opt.track:
                    opt.show_track = True

                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    tracks = sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        dataset_recognised_names = detect_person_in_video(im0)
                        for i, name in enumerate(dataset_recognised_names):
                            if detected_dataset_names[i] == 'Unknown':
                                detected_dataset_names[i] = name

                        if opt.show_track:
                            # loop over tracks
                            for t, track in enumerate(tracks):
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else \
                                    sort_tracker.color_list[t]
                                # threshold = int(750 / 1080 * 856)
                                threshold = 750
                                isSavePhoto = False
                                isExit = True
                                # print(t, 't')
                                currentDatasetName = detected_dataset_names[t]
                                if int(track.centroidarr[len(track.centroidarr) - 1][1]) < threshold and int(
                                        track.centroidarr[0][1]) > threshold:
                                    del tracks[t]
                                    detected_dataset_names[t] = 'Unknown'
                                    if cameraType == 'entrance':
                                        action = 'exit'
                                    else:
                                        action = 'entrance'
                                    isSavePhoto = True
                                if int(track.centroidarr[len(track.centroidarr) - 1][1]) > threshold and int(
                                        track.centroidarr[0][1]) < threshold:
                                    del tracks[t]
                                    detected_dataset_names[t] = 'Unknown'
                                    if cameraType == 'entrance':
                                        action = 'entrance'
                                    else:
                                        action = 'exit'
                                    isSavePhoto = True
                                if isSavePhoto and action == cameraType:
                                    photoName = str(uuid.uuid4())
                                    save_photo_url = save_photo_dir + photoName + '.jpg'
                                    cv2.imwrite(save_photo_url, im0)
                                    data = {'image': save_photo_url, 'action': action, 'camera': ip[0],'name_file': currentDatasetName}
                                    print(data, 'data')
                                    # date_now = datetime.today()
                                    # nowSeconds = date_now.timestamp()
                                    # difference = nowSeconds - startTimeSeconds
                                    #
                                    # print(frames / difference, 'fps')
                                    # r = requests.post(url = 'http://django:8000/api/employees/history/', json = data)
                                    # res = r.json()
                                    # print(res, 'res')

                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                        min_hits=2,
                        iou_threshold=0.2)

    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
