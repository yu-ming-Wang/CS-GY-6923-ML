import cv2
import os
import zipfile
from pathlib import Path

def frame_sample(video, save_dir):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Failed to open video: {video}")
        return

    file_name = Path(video).stem
    try:
        save_path = Path(save_dir).joinpath(file_name)
        if not save_path.exists():
            save_path.mkdir()
    except OSError:
        print('Error: Creating directory of data')

    cap.set(cv2.CAP_PROP_FPS, 25)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 6 * 5)
    count = 0

    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        
        if frame is None:
            continue
        frame = cv2.resize(frame, (456, 256), interpolation=cv2.INTER_CUBIC)
        name = f"{save_path}/frame_{str(count)}.jpg"
        cv2.imwrite(name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def crop_to_square(img):
    h, w, _ = img.shape
    c_x, c_y = int(w / 2), int(h / 2)
    img = img[:, c_x - c_y: c_x + c_y]
    return img

# def frame_extract(video_path, save_dir, resize=(456, 256), transform=None):
#     cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
#     file_name = Path(video_path).stem

#     save_path = Path(save_dir).joinpath(file_name)
#     os.makedirs(save_path, exist_ok=True)

#     length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     count = 0

#     while cap.isOpened():
#         count += 1
#         if length == count:
#             break
        
#         ret, frame = cap.read()
#         if frame is None:
#             continue
        
#         if transform is not None:
#             frame = transform(frame)
        
#         frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
#         name = f"{str(save_path)}/frame_{str(count)}.jpg"
#         cv2.imwrite(name, frame)

def frame_extract(video_path, save_dir, resize=(456, 256), transform=None):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    file_name = Path(video_path).stem

    save_path = Path(save_dir).joinpath(file_name)
    os.makedirs(save_path, exist_ok=True)

    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0

    while cap.isOpened():
        count += 1
        if length == count:
            break
        
        ret, frame = cap.read()
        if frame is None:
            continue
        
        if transform is not None:
            frame = transform(frame)
        
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
        name = f"{str(save_path)}/frame_{str(count)}.jpg"
        cv2.imwrite(name, frame)
    
    cap.release()
    return str(save_path)

def long_time_task(video, parent_dir):
    print(f"execute {video} ...")
    return frame_extract(video_path=video, save_dir=parent_dir, resize=(256, 256), transform=crop_to_square)

if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='extract image frames from videos')
    parser.add_argument('-v', '--video-dir', help="path to video directory", default=None, type=str)
    parser.add_argument("-o", "--output-dir", default=None, type=str, help="path to the extracted frames")
    args = parser.parse_args()

    p = Pool(8)
    v_path = args.video_dir
    path = Path(v_path)
    video_pts = list(path.rglob("*.mp4"))

    for video in tqdm(video_pts):
        video_path = str(video)
        if args.output_dir is not None:
            saved_dir = args.output_dir
        else:
            saved_dir = Path(video).parent
        p.apply_async(long_time_task, args=(video_path, saved_dir))

    p.close()
    p.join()
    print('All subprocesses done.')
