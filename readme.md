# eg
**粗体**
-------------------------
*斜体*
## 欢迎来到[梵居闹市](http://blog.leanote.com/freewalk)
<font color=red size=6 face ="黑体">红色</font>

# Github 
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






# 3d模型训练的思路：
###  选用3d模型训练的时候，主要要依赖于monai的框架，它主要处理的领域是医学图像处理，在数据集的制作时，是将图像提前在深度上堆叠然后再加上一个通道后保存为nii.gz的格式方便后面用monai去读取，选用的模型也是3d的unet模型，注意的是深度方向不能太小，要保证下采样结束之后深度仍然大于1.在验证的时候主要采用的是monai中sliding_window_inference这个方法。但是目前预测的结果很差
