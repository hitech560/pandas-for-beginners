"""
《python数据分析基础》之描述性统计与建模

数据集
    红葡萄酒数据集：
    http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

    白葡萄酒数据集：
    http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

红葡萄酒文件中包含1599条观测，白葡萄酒文件包含4898条观测。输入变量是葡萄酒的物理化学成分和特性，
包括非挥发性酸、挥发性酸、柠檬酸、残余糖分、氯化物、游离二氧化硫、总二氧化硫、密度、pH值、硫酸盐
和酒精含量。我们将两个文件合并，并增加一列type变量，用于区分白葡萄酒和红葡萄酒：

分别打印红葡萄酒和白葡萄酒的摘要统计量:
       quality
        count      mean       std  min  25%  50%  75%  max 
type                                                       
red    1599.0  5.636023  0.807569  3.0  5.0  6.0  6.0  8.0 
white  4898.0  5.877909  0.885639  3.0  5.0  6.0  6.0  9.0 

分别打印红葡萄酒和白葡萄酒的摘要统计量(结果重排列):
                 type 
quality  count  red      1599.000000
                white    4898.000000
         mean   red         5.636023
                white       5.877909
         std    red         0.807569
                white       0.885639
         min    red         3.000000
                white       3.000000
         25%    red         5.000000
                white       5.000000
         50%    red         6.000000
                white       6.000000
         75%    red         6.000000
                white       6.000000
         max    red         8.000000
                white       9.000000
dtype: float64

分别计算第25百分位数和第75百分位数(结果重排列):
      quality
type     red white
0.25     5.0   5.0
0.75     6.0   6.0

分组计算标准差:
         quality
            std
type
red    0.807569
white  0.885639
"""

# 导入库
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
red_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases"
           "/wine-quality/winequality-red.csv"
           )
white_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases"
             "/wine-quality/winequality-white.csv"
             )
red_w = pd.read_csv(red_url, sep=";", header=0)
white_w = pd.read_csv(white_url, sep=";", header=0)

# 添加类型列
red_w.insert(0, 'type', 'red')
white_w.insert(0, 'type', 'white')

# 合并红葡萄酒和白葡萄酒数据
wine = pd.concat([red_w, white_w])

# 重命名列名
wine.columns = wine.columns.str.replace(' ', '_')

#按照葡萄酒类型显示质量的描述性统计量，分别打印红葡萄酒和白葡萄酒的摘要统计量
print('分别打印红葡萄酒和白葡萄酒的摘要统计量:\n',wine.groupby('type')[['quality']].describe())

#unstack函数将结果重新排列，使统计量显示在并排的两列中
print('分别打印红葡萄酒和白葡萄酒的摘要统计量(结果重排列):\n',
      wine.groupby('type')[['quality']].describe().unstack('type')
      )

#按照葡萄酒类型显示质量的特定分位数值，quantile函数对质量列计算第25百分位数和第75百分位数
print('分别计算第25百分位数和第75百分位数(结果重排列):\n',
      wine.groupby('type')[['quality']].quantile([0.25, 0.75]).unstack('type')
      )

#检验红葡萄酒和白葡萄酒的平均质量是否有所不同，分组计算标准差
print('分组计算标准差:\n',wine.groupby(['type'])[['quality']].agg(['std']))

# 按类型提取质量数据
red_wine = wine.loc[wine['type'] == 'red', 'quality']
white_wine = wine.loc[wine['type'] == 'white', 'quality']

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制直方图
# 注意：density=True 用于显示密度，alpha 设置透明度
plt.hist(red_wine, bins=20, density=True, color='red', alpha=0.5, label='Red wine')
plt.hist(white_wine, bins=20, density=True, color='blue', alpha=0.5, label='White wine')

# 设置轴标签和标题
plt.xlabel('Quality Score')
plt.ylabel('Density')
plt.title('Distribution of Quality by Wine Type')
plt.legend()

# 显示图像
plt.show()
