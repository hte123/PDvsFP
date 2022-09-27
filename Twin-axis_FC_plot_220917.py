'''
Author: Youqi Liu

2022.02.19 edited by Tianen He 
1. change the colors, FP: #ffb300 (orange), PD: #1e88e5 (blue), overlap: #41B865 (green)
2. use symbol intead of uniprot ID to mark the proteins
3. each color represents a category of deps, mark the number of deps of each category in the legend

2022.03.11 edited by Tianen He 
change the legend texts

'''

#%% 

from math import log2
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


lcdict={
    'up,-':'FP-unique DEP,#ffb300',
    'down,-':'FP-unique DEP,#ffb300',
    '-,up':'PD-unique DEP,#1e88e5',
    '-,down':'PD-unique DEP,#1e88e5',
    'up,up':'Overlapped DEP,#41B865',
    'down,down':'Overlapped DEP,#41B865',
    'up,down':'FP-up and PD-down DEP,red',
    'down,up':'FP-down and PD-up DEP,red',
    '-,-':'Non-DEP,#828282'
}

    
def Twin_fc_plot(data1,data2,axis_labels,lcdict):
    comb=pd.merge(data1,data2,on='protein').set_index('protein')
    comb['color']=comb.apply(lambda row:lcdict[row['threshold_x']+','+row['threshold_y']].split(','),axis=1)
    comb['label']=comb['color'].str[0]
    comb['color']=comb['color'].str[1]

    counts = pd.DataFrame(comb['label'].value_counts())
    counts['label'] = counts.index.values + " (" + counts['label'].map(str) + ")"
    comb['label'] = comb.apply(lambda row: counts.loc[row['label'], 'label'], axis=1)

    fig,ax=plt.subplots(figsize=(8,8))
    for label in comb['label'].unique():
        tmp=comb.loc[comb['label']==label,:]
        color=tmp['color'].unique()[0]
        plt.scatter(tmp['logFC_x'],tmp['logFC_y'],label=label,color=color,s=30)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.axvline(0,color='gray')
    plt.axhline(0,color='gray')
    plt.legend(frameon=False,bbox_to_anchor=(1,1))
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])

    for i in range(0, comb.shape[0]):
        row = comb.iloc[i]

        thres = log2(3)
        if row['color'] == 'red' or (row['logFC_x'] > thres and row['logFC_y'] > thres) or (row['logFC_x'] < -thres and row['logFC_y'] < -thres):
            plt.text(row['logFC_x'], row['logFC_y'], row['symbol_x'], color = row['color'])

axis_labels=['FP log2(fold change)','PD log2(fold change)']
fp_cortex=pd.read_csv('volcano_FP_cortex.tsv',sep='\t').loc[:,['logFC','protein','threshold', 'symbol']]
pd_cortex=pd.read_csv('volcano_PD_cortex.tsv',sep='\t').loc[:,['logFC','protein','threshold', 'symbol']]
Twin_fc_plot(fp_cortex,pd_cortex,axis_labels,lcdict)
plt.savefig('Figure4B_volcano_cortex.pdf',dpi=600,bbox_inches='tight')

fp_medulla=pd.read_csv('volcano_FP_medulla.tsv',sep='\t').loc[:,['logFC','protein','threshold', 'symbol']]
pd_medulla=pd.read_csv('volcano_PD_medulla.tsv',sep='\t').loc[:,['logFC','protein','threshold', 'symbol']]
Twin_fc_plot(fp_medulla,pd_medulla,axis_labels,lcdict)
plt.savefig('Figure4B_volcano_medulla.pdf',dpi=600,bbox_inches='tight')

# %%
