"""
对标签文件重新定义其结构
"""
import json


def relabel(path, save_dir):
    new_label = {}
    with open(path) as f:
        # index = 0
        for line in f:
            lst = [0] * 63
            temp = line.split(",")
            name = temp[0].split(".")[0]
            raw_label = temp[1:]
            for i in raw_label:
                lst[int(i)] = 1
            new_label[name] = lst
            # index += 1
            # if index % 100:
            #     print("第%d个视频label已处理" % index)
    # print("%d个视频label全部处理完毕" % index)

    with open(save_dir, 'w') as f:
        f.write(json.dumps(new_label, indent=0))


if __name__ == '__main__':
    relabel(r"D:\data\video classification\train_set\readme\short_video_trainingset_annotations.txt",
            r"D:\data\video classification\train_set\train_label.json")
    relabel(r"D:\data\video classification\validation_set\readme\short_video_validationset_annotations.txt",
            r"D:\data\video classification\validation_set\validation_label.json")
