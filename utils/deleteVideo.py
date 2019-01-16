"""
除去没有label的视频
"""
import os
# import shutil


def deletelabel(video_path, label_path):
    # 所有视频name
    all_video = [file.split('.')[0] for file in os.listdir(video_path)]

    # 所有有label的视频name
    all_label = []
    with open(label_path) as f:
        for line in f:
            all_label.append(line.split(",")[0].split(".")[0])

    extra_video = set(all_video) - set(all_label)
    for file in all_video:
        if file not in extra_video:
            continue
        dele_path = os.path.join(video_path, str(file) + ".mp4")
        print("删除文件 " + dele_path)
        os.remove(dele_path)
    print("共删除文件%d份" % len(extra_video))


if __name__ == "__main__":
    deletelabel(r"D:\data\video_classification\train_set\train_video",
                r"D:\data\video_classification\train_set\readme\short_video_trainingset_annotations.txt")
    deletelabel(r"D:\data\video_classification\validation_set\validation_video",
                r"D:\data\video_classification\validation_set\readme\short_video_validationset_annotations.txt")
