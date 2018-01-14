# @Time    : 2017/12/27 19:25
# @Author  : myailab
# @Site    : www.myailab.cn
# @File    : read_lenna.py
# @Software: PyCharm

from matplotlib import pyplot as plt
import mahotas as mh

image = mh.imread("H:\Python Workspace\BuildingMLSys\chapter_10\Lenna.png")
image = image - image.mean()
plt.imshow(image)
plt.show()