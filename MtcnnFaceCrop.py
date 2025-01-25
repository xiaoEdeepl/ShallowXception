from mtcnn import MTCNN
import cv2
import os
import multiprocessing as mp


def process_video(video_file, frame_number, confidence, input_folder, output_root_folder):
    detector = MTCNN()  # 每个进程中初始化一次 MTCNN
    video_path = os.path.join(input_folder, video_file)
    print(f"正在处理视频：{video_path}")
    video_folder = os.path.join(output_root_folder, os.path.splitext(video_file)[0])

    # 检查视频是否已经处理过
    if os.path.exists(video_folder):
        print(f"视频 {video_file} 已处理，跳过...")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return  # 如果无法打开视频，跳过

    os.makedirs(video_folder, exist_ok=True)

    frame_count = 0
    while frame_count < frame_number:
        ret, frame = cap.read()
        if not ret:
            print('视频读取错误')
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)

        for i, result in enumerate(results):
            if result['confidence'] < confidence:
                continue

            keypoints = result['keypoints']
            left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']
            nose = keypoints['nose']

            # 提取人脸框
            x, y, w, h = result['box']
            x, y = max(0, x), max(0, y)
            cropped_face = frame[y:y+h, x:x+w]

            # 保存裁剪的人脸
            output_path = os.path.join(video_folder, f"frame_{frame_count}_face_{i}.jpg")
            cv2.imwrite(output_path, cropped_face)

        frame_count += 1

    cap.release()  # 释放视频文件

def process_videos_in_parallel(video_files, input_folder, output_root_folder, frame_number, confidence):
    # 使用多进程处理视频文件
    with mp.Pool(processes=8) as pool:
        pool.starmap(process_video, [(video_file, frame_number, confidence, input_folder, output_root_folder) for video_file in video_files])

if __name__ == '__main__':
    input_folder = 'D:\\dfdc_train_part_2'
    output_root_folder = './dataset/dfdc'
    os.makedirs(output_root_folder, exist_ok=True)

    # 过滤出视频文件，确保是有效的格式
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    # 使用多进程处理所有视频
    process_videos_in_parallel(video_files, input_folder, output_root_folder, 25, 0.95)

    print("所有视频处理完成！")
