from matplotlib import pyplot as plt

labels = ['Background Image','Checked View','Icon','Input Field',
          'Image','Text','Text Button','Page Indicator',
          'Pop-Up Window','Sliding Menu','Switch']
mAP_VOC = [0.9233, 0.3751, 0.7592, 0.6979, 0.8200, 0.8418, 0.9331, 
           0.7512, 0.8260, 0.9796, 0.9357]
mAP_COCO = [0.8460, 0.1687, 0.3870, 0.3284, 0.5216, 0.4272, 0.5896, 
            0.3096, 0.5684, 0.8406, 0.6044]

fig, ax = plt.subplots(figsize=(24,9))
voc = ax.barh(labels, mAP_VOC)
coco = ax.barh(labels, mAP_COCO)
#ax.barh(labels, mAP_COCO)

#for s in ['top','bottom']:
#    ax.spines[s].set_visible(False)

ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10, labelsize=15)
 
# Add x, y gridlines
ax.grid(color='grey',linestyle='-.',linewidth=0.5,alpha = 0.5)
 
# Show top values 
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width() - 0.05, i.get_y()+0.6, 
             str(round((i.get_width()), 2)),
             fontsize = 20, fontweight ='bold',
             color ='white')

ax.set_xlim([0,1])
ax.set_xlabel('mAP',fontsize=30)
ax.set_ylabel('Class (Interactable Label)',fontsize=30)

ax.legend((voc,coco),('mAP@0.5','mAP@[0.5,0.95]'),fontsize=30, loc=3)

# Add Plot Title
ax.set_title('Class-based mAP for Web350k-VINS',fontsize=40)

plt.savefig('./web350kVINS_plot.pdf')
