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
import asyncio
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

host_ip = os.environ.get['HOST_IP']
print(host_ip, 'host_ip')
if not host_ip:
    host_ip = 'localhost'
# docker
cameraUrls = os.environ['CAMERA_URLS']
cameraTypes = os.environ['CAMERA_TYPES']

# dataset_names = []
# dataset = os.walk("dataset")
# known_face_encodings = []
# for data in dataset:
#     dataset_names = data[2]
#     for dataset_name in data[2]:
#         loaded_dataset = pickle.loads(open("dataset/" + dataset_name, "rb").read())
#         print(dataset_name, 'dataset_name')
#         known_face_encodings.append(loaded_dataset)
# print(dataset_names, 'dataset_names')
# def detect_person_in_video(image):
#     rframe = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
#     locations = face_recognition.face_locations(rframe)
#     if not len(locations):
#         return [[], image]
#     img = image
#     for face_location in locations:
#         img = cv2.circle(image, (face_location[1] * 4,  face_location[0] * 4), 5, (255,0,0), thickness=1, lineType=8, shift=0)
#         img = cv2.circle(img, (face_location[3] * 4,  face_location[2] * 4), 5, (0, 0, 255), thickness=1, lineType=8, shift=0)
#     encodings = face_recognition.face_encodings(rframe, locations)
#     datasets_matches = []
#     for id, face_encoding in enumerate(encodings):
#         matches = face_recognition.face_distance(known_face_encodings, face_encoding)
#         datasets_matches.append(matches)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         # bottomLeftCornerOfText = (locations[id][1], locations[id][2])
#         fontScale = 1
#         fontColor = (0, 255, 0)
#         thickness = 1
#         lineType = 2
#         # cv2.putText(img, 'Hello World!',
#         #             bottomLeftCornerOfText,
#         #             font,
#         #             fontScale,
#         #             fontColor,
#         #             thickness,
#         #             lineType)
#     print(datasets_matches, 'datasets_matches')
#     return [datasets_matches, img]

# detected_dataset_names = ['unknown'] * 30

def detect():
    sources, actions, weights, imgsz = opt.sources, opt.actions, opt.weights, opt.img_size
    # docker
    if not sources:
        sources = cameraUrls.split(' ')
        actions = cameraTypes.split(' ')

    if sources[0] == '0':
        save_photo_dirs = ['images/123/', 'images/ip_camera/']
        if not os.path.exists(save_photo_dirs[0]):
            os.mkdir(save_photo_dirs[0])
        if not os.path.exists(save_photo_dirs[1]):
            os.mkdir(save_photo_dirs[1])
        ips = ['123', 'ip_camera']
        frames_counters = [0, 0]
        tracksImages = [{}, {}]
    else:
        ips = []
        frames_counters = []
        save_photo_dirs = []
        tracksImages = []
        for source in sources:
            print(source)
            ip_address = re.findall(r'(?:\d{1,3}\.)+(?:\d{1,3})', source)[0]
            ips.append(ip_address)
            dir_name = 'images/' + ip_address + '/'
            save_photo_dirs.append(dir_name)
            frames_counters.append(0)
            tracksImages.append({})
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    trace = True
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
    dataset = LoadStreams(sources=sources, img_size=imgsz, stride=stride)

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
    ###################################
    startTime = 0
    frames_counter_fps = 0
    for path, img, im0s, vid_cap in dataset:

        currentTime = time.time()
        fps = 1 / (currentTime - startTime)
        startTime = currentTime
        if frames_counter_fps == 300:
            print(fps, 'FPS')
            frames_counter_fps = 0
        frames_counter_fps += 1
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

                    tracked_dets = sort_trackers[i].update(dets_to_sort, opt.unique_track_color)
                    tracks = sort_trackers[i].getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets) > 0:
                        photoSavedUrl = False
                        if frames_counters[i] == 50:
                            photoName = str(uuid.uuid4())
                            save_photo_url = save_photo_dirs[i] + photoName + '.jpg'
                            photoSavedUrl = save_photo_url
                            cv2.imwrite(save_photo_url, im0)
                            frames_counters[i] = 0
                        if frames_counters[i] != 50:
                            frames_counters[i] += 1

                        # bbox_xyxy = tracked_dets[:, :4]
                        # identities = tracked_dets[:, 8]
                        # categories = tracked_dets[:, 4]
                        # confidences = None
                        #
                        # detected_persons = detect_person_in_video(im0)
                        # dataset_recognised_matches = detected_persons[0]
                        dataset_recognised_matches = []
                        # im0 = detected_persons[1]
                        # for matchId, matches in enumerate(dataset_recognised_matches):
                        #     if detected_dataset_names[matchId] == 'unknown':
                        #         detected_dataset_names[matchId] = [matches]
                        #     else:
                        #         detected_dataset_names[matchId].append(matches)
                        if opt.show_track:
                            # loop over tracks
                            for t, track in enumerate(tracks):
                                x_coord = int(track.centroidarr[len(track.centroidarr) - 1][0])
                                y_coord = int(track.centroidarr[len(track.centroidarr) - 1][1])
                                try:
                                    if not tracksImages[i][t]:
                                        tracksImages[i][t] = []
                                except KeyError:
                                    tracksImages[i][t] = []
                                if photoSavedUrl:
                                    tracksImages[i][t].append({'image': photoSavedUrl, 'body_coordinates': {'x': x_coord, 'y': y_coord}})
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else \
                                sort_trackers[i].color_list[t]

                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])),
                                          (int(track.centroidarr[i + 1][0]),
                                           int(track.centroidarr[i + 1][1])),
                                          track_color, thickness=opt.thickness)
                                 for i, _ in enumerate(track.centroidarr)
                                 if i < len(track.centroidarr) - 1]
                                thresholds = [750, 750]
                                threshold = thresholds[i]
                                cv2.line(im0, (0, threshold), (1919, threshold), (0, 0, 255), 2)
                                isSavePhoto = False
                                cameraType = actions[i]
                                if y_coord < threshold and int(
                                        track.centroidarr[0][1]) > threshold:
                                    del tracks[t]
                                    temp_tracks_images = tracksImages[i][t]
                                    tracksImages[i][t] = []
                                    if cameraType == 'entrance':
                                        action = 'exit'
                                    else:
                                        action = 'entrance'
                                    isSavePhoto = True
                                if y_coord > threshold and int(
                                        track.centroidarr[0][1]) < threshold:
                                    del tracks[t]
                                    temp_tracks_images = tracksImages[i][t]
                                    tracksImages[i][t] = []
                                    if cameraType == 'entrance':
                                        action = 'entrance'
                                    else:
                                        action = 'exit'
                                    isSavePhoto = True
                                if isSavePhoto and action == cameraType:
                                    # if detected_dataset_names[t] == 'unknown':
                                    #     currentDatasetName = 'unknown'
                                    # else:
                                    #     averageMatches = []
                                    #     faceFrames = 0
                                    #     for currentMatch in detected_dataset_names[t]:
                                    #         if currentMatch == 'unknown':
                                    #             continue
                                    #         faceFrames += 1
                                    #         if not len(averageMatches):
                                    #             averageMatches = currentMatch
                                    #         else:
                                    #             for matchId, eachMatch in enumerate(currentMatch):
                                    #                 averageMatches[matchId] += eachMatch
                                    #     for averageMatchId, averageMatch in enumerate(averageMatches):
                                    #         averageMatches[averageMatchId] = averageMatch / faceFrames
                                    #     best_match_index = np.argmin(averageMatches)
                                    #     currentDatasetName = dataset_names[best_match_index]
                                    # detected_dataset_names[t] = 'unknown'
                                    if not photoSavedUrl:
                                        photoName = str(uuid.uuid4())
                                        save_photo_url = save_photo_dirs[i] + photoName + '.jpg'
                                        im0 = cv2.circle(im0,
                                                         (int(track.centroidarr[0][0]), int(track.centroidarr[0][1])),
                                                         5,
                                                         (255, 0, 0), thickness=1, lineType=8, shift=0)
                                        cv2.imwrite(save_photo_url, im0)
                                        temp_tracks_images.append({'image': save_photo_url, 'body_coordinates': {'x': x_coord, 'y': y_coord}})
                                    else:
                                        im0 = cv2.circle(im0,
                                                         (int(track.centroidarr[0][0]), int(track.centroidarr[0][1])),
                                                         5,
                                                         (255, 0, 0), thickness=1, lineType=8, shift=0)
                                        cv2.imwrite(photoSavedUrl, im0)
                                    face_frames = []

                                    for face_frame in temp_tracks_images:
                                        face_frames.append({'image': face_frame['image'], 'body_coordinates': face_frame['body_coordinates']})
                                    data = {'frames': face_frames, 'action': action, 'camera_address': ips[i]}
                                    print(data, 'data')
                                    print('frames: ', len(face_frames))
                                    r = requests.post(url = 'http://' + host_ip + ':8008/action', json = data)
            # cv2.imshow(str(p), im0)
            # cv2.waitKey(1)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov72.pt', help='model.pt path(s)')
    parser.add_argument('--sources', nargs='+', type=str, default=False, help='sources')
    parser.add_argument('--actions', nargs='+', type=str, default=False, help='actions')
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
    print(opt.sources)
    np.random.seed(opt.seed)

    sources_array = cameraUrls.split(' ')
    if not sources_array:
        sources_array = ['0']
    sort_trackers = []
    for source in sources_array:
        tracker = Sort(max_age=5,
                       min_hits=2,
                       iou_threshold=0.2)
        sort_trackers.append(tracker)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov72.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
