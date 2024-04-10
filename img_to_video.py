import cv2
import os


def write_video(type,video_num):

    # 資料夾路徑
    folder_path = f'saved_video/{type}/{video_num}'

    # 影片儲存路徑及檔名
    output_video = f'saved_video/{type}.mp4'

    # 取得資料夾內所有圖片的檔案名稱
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])

    # print(image_files)
    # 讀取第一張圖片以獲取寬度和高度
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, _ = first_image.shape

    # 設定影片編碼器、FPS、影片大小
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 逐一讀取圖片並寫入影片
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # 釋放影片寫入器
    video_writer.release()

    print('影片已儲存至', output_video)

if __name__ == '__main__':

    model_type = [
        'gt',
        'epipolar_epoch470_inter1_samestart',
        'mae025_epoch210_inter1_randstart',
                  ]
    video_num = 25

    for type in model_type:
        write_video(type,video_num)