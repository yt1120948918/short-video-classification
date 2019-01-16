"""
该py文件下均是系统需要用到的一些工具函数
"""


def print_time(start_time, end_time, des="整个过程"):
    spend_time = end_time - start_time
    if spend_time < 60:
        print(des + "耗时 %.2f 秒" % spend_time)
    elif 60 <= spend_time < 3600:
        print(des + "耗时 %.2f 分钟" % (spend_time / 60))
    elif spend_time >= 3600:
        print(des + "耗时 %.2f 小时" % (spend_time / 3600))
