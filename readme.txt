/dataset目录：存储实验数据
    train/test.csv      每个台风ID合成list的条目
    train19/test19.csv  固定宽度19的滑动窗口截取的MaxWind时间序列数据
（测试数据选取2014和2015年的pacific台风）

/images目录：保存实验截图
    pip_list.PNG 放假前用于记录pip list各python包及版本
/venv目录：python虚拟环境（没用？）

Model.py 核心代码
Preprocessing.py 数据预处理，用于把从kaggle下载下来的NOAA的raw data转化成train/test.csv格式数据
test.py 用于临时测试

readme.txt 本文档

------------------------------------------------------------------------------------------------------------------------
# trian/test data list index:
    # 0 ID
    # 1 Date
    # 2 Time
    # 3 Event
    # 4 Status
    # 5 Latitude
    # 6 Longitude
    # 7 MaxWind
