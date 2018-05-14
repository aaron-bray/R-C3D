import sys, os, errno
import numpy as np
import csv
import json
import copy
import math

#assert len(sys.argv) == 2, "Usage: python log_analysis.py <test_log>"
logfile = sys.argv[1]
FPS = 25.0   # currently FPS=25
ACT_ID = 0

def generate_classes():
    classes = {0: 'Background',
               1: 'vehicle_u_turn',
               2: 'Closing_Trunk',
               3: 'Open_Trunk',
               4: 'Loading',
               5: 'Unloading',
               6: 'Transport_HeavyCarry',
               7: 'Entering',
               8: 'Exiting',
               9: 'Opening',
               10: 'Closing',
               11: 'vehicle_turning_left',
               12: 'vehicle_turning_right'}
    return classes

classes = generate_classes()

def get_segments(data, thresh):
    segments = []
    vid = 'Background'
    find_next = False
    global ACT_ID

    for l in data:
      # video name and sliding window length
      if "fg_name :" in l:
         vid = l.split('/')[-1] #[4]

      # frame index, time, confident score
      elif "frames :" in l:
         start_frame=int(l.split()[4])
         end_frame=int(l.split()[5])
         stride = int(l.split()[6].split(']')[0])

      elif "activity:" in l:
         label = int(l.split()[1])
         find_next = True

      elif "im_detect" in l:
         return vid, segments

      elif find_next: #Next temporal segment if it exists
         lc= l.strip()
         la= lc.strip('[')
         lb= la.strip(']')

         left_frame = float(lb.split()[0])*stride + start_frame
         right_frame = float(lb.split()[1])*stride + start_frame
         if (left_frame < end_frame) and (right_frame <= end_frame):
           left  = left_frame / FPS
           right = right_frame / FPS
           score = float(lb.split()[2].split(']')[0])
         elif (left_frame < end_frame) and (right_frame > end_frame):
             if (end_frame-left_frame)*1.0/(right_frame-left_frame)>=0:
                 right_frame = end_frame
                 left  = left_frame / FPS
                 right = right_frame / FPS
                 score = float(lb.split()[2].split(']')[0])
         if score > thresh:
           ACT_ID = ACT_ID + 1
           if label not in classes:
             classes[label] = "out_of_bounds"
           activity = {'activity' : classes[label], 'activityID': ACT_ID}
           frames = {math.floor(left) : 1, math.floor(right) : 0}
           activity['localization'] = { vid : frames}
           segments.append(activity)


def analysis_log(logfile, thresh):
  with open(logfile, 'r') as f:
    lines = f.read().splitlines()
  
  predict_data = []
  res = {}
  for l in lines:
    if "frames :" in l:
      predict_data = []
    predict_data.append(l)
    if "im_detect:" in l:
      vid, segments = get_segments(predict_data, thresh)
      if vid not in res:
        res[vid] = []
      res[vid] += segments

  return res

segmentations = analysis_log(logfile, thresh = 0.05)

for vid, vinfo in segmentations.iteritems():
  vid_res = {'filesProcessed': [ vid ], 'activities': vinfo }
  outfile = open(vid + '.json', 'w')
  json.dump(vid_res, outfile)