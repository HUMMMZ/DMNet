import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import sys

# TN3K, TNUI
preds_dir = 'visualization/TNUI/'
preds = [preds_dir + i for i in os.listdir(preds_dir) if i.endswith("npy")]
#print(preds)

plt.rcParams.update({'font.size': 12}) 
plt.rc('font',family='Times New Roman')
plt.figure()
color_list = ['saddlebrown', 'blue', 'gold', 'green', 'black', 'orange', 'red']
if 'TNUI' in preds_dir:
    true_masks = np.load('visualization/tnui_mask_test.npy')  # 真实的分割掩码
else:
    true_masks = np.load('visualization/tn3k_mask_test.npy')
true_masks = true_masks/255.
true_masks = np.where(true_masks>=0.5, 1, 0)
true_masks_flat = true_masks.flatten()
j = 0

for i in preds:
    label_name = i.split('.')[0].split('1')[1]
    print(label_name)
    # 加载预测结果和真实标签
    pred_masks = np.load(i)
    # 移除pred_masks中不必要的维度
    pred_masks = pred_masks.squeeze(axis=1)
    # Flatten 掩码以便于处理
    pred_masks_flat = pred_masks.flatten()
    # 计算精确率和召回率
    precision, recall, thresholds = precision_recall_curve(true_masks_flat, pred_masks_flat)
    # 计算AUC
    auc_score = str("{:.2f}".format(auc(recall, precision)))
    label_name = label_name+'(AUC='+auc_score+')'
    # 绘制PR曲线
    plt.plot(recall, precision, color_list[j], label=label_name)
    j = j + 1


plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.title('PR Curve for fcn, Model')
plt.legend(loc="lower left")
plt.savefig(preds_dir + 'PR.png')
