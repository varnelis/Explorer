from matplotlib import pyplot as plt
import matplotlib

labels = ['StaticText','link','listitem','paragraph','heading','img']
mAP_VOC = [0.0617, 0.0850, 0.1080, 0.0067, 0.0427, 0.1305]
mAP_COCO = [0.0153, 0.0245, 0.0322, 0.0018, 0.0148, 0.0414]

fig, ax = plt.subplots(figsize=(24,9))
voc = ax.barh(labels, mAP_VOC)
coco = ax.barh(labels, mAP_COCO)
#ax.barh(labels, mAP_COCO)

#matplotlib.rcParams.update({'font.size':25})

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
'''
patch = set()
xdist = {}
for i in ax.patches:
    xy = i.get_xy()
    if xy in patch:
        ec='orange'
    else:
        patch.add(xy)
        xdist[xy] = i.get_width() + 0.01
        ec='blue'
    bbox = dict(boxstyle='round',fc='blanchedalmond',ec=ec,alpha=1)
    plt.text(xdist[xy], i.get_y()+0.65 if ec=='orange' else i.get_y()+0.25, 
             str(round((i.get_width()), 4)),
             fontsize = 10, bbox=bbox, fontweight ='bold',
             color = ec)
'''
c = 0
for i in ax.patches:
    if c != 0 and c != 6 and c%3==0:
        c+=1
        continue
    c += 1
    plt.text(i.get_width() - 0.01, i.get_y()+0.5, 
             str(round((i.get_width()), 2)),
             fontsize = 20, fontweight ='bold',
             color ='white')


ax.set_xlim([0,0.2])
ax.set_xlabel('mAP',fontsize=16)
ax.set_ylabel('Class (Interactable Label)',fontsize=30)

ax.legend((voc,coco),('mAP@0.5','mAP@[0.5,0.95]'),fontsize=30)

# Add Plot Title
ax.set_title('Class-based mAP for Web350k',fontsize=40)

plt.savefig('./web350k_plot.pdf')
