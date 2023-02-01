#!/bin/bash
pwd &&
python -u detect_or_track_save_and_send.py --weights yolov7.pt --classes 0 --nosave --track
