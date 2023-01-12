#!/bin/bash
pwd &&
python -u detect_or_track_save_and_send.py --weights yolov7.pt --classes 0 --source rtsp://admin:just4Taqtile@192.168.1.161/Streaming/Channels/1 --nosave --track
#python detect_heads.py --weights yolov7.pt --classes 0 --source rtsp://admin:just4Taqtile@192.168.1.64/h264_stream --nosave
#python detect_heads.py --weights yolov7.pt --classes 0 --source rtsp://admin:just4Taqtile@192.168.1.160/Streaming/Channels/1 --nosave
#python detect_heads.py --weights yolov7.pt --classes 0 --source rtsp://admin:just4Taqtile@192.168.1.161/Streaming/Channels/1 --nosave
