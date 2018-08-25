import argparse
import cv2
import os
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='command for convert video to image for ucf101')
    parser.add_argument('--video-path', type=str, default='', help='the video path')
    parser.add_argument('--save-path', type=str, default='', help='saving path')
    args = parser.parse_args()

    action_list = os.listdir(args.video_path)
    pbdr = tqdm(total=len(action_list))
    for action in action_list:
        pbdr.update(1)
        if not os.path.exists(args.save_path+action):
            os.mkdir(args.save_path+action)
        video_list = os.listdir(args.video_path+action)
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(args.save_path+action+'/'+prefix):
                os.mkdir(args.save_path+action+'/'+prefix)
            save_name = args.save_path + action + '/' + prefix + '/'
            video_name = args.video_path+action+'/'+video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(save_name+str(10000+fps_count)+'.jpg', frame)
                    fps_count += 1
            cap.release()
    pbdr.close()
