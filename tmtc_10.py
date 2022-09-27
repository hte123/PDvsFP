#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tmtc.py
@Time    :   2021/11/24 15:43:49
@Author  :   Youqi Liu
@Version :   1
@Contact :   liuyouqi@westlakeomics.com

edited by Tianen He
'''

#%% ##########Function definition and import packages
import pandas as pd
import os
from scipy.stats import ks_2samp

os.environ['R_HOME'] = 'D:/Program Files/R-4.2.1' #set your R path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn import preprocessing as sp
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["pdf.fonttype"] = 42

def to_percent(temp,position):
    return '%1.0f' % (100*temp) + '%'

# cv
def cv(x):
        return x.std() / x.mean()

# jointplot figure2b
def jointplot(mat1,mat2,name,labels,text=''):
    df1=pd.DataFrame([mat1.agg(cv,axis=1),mat1.agg(np.nanmean,axis=1)],
                    index=['Coefficient of variation','Average of abundance '+text]).T
    df1['method']=labels[0]

    df2=pd.DataFrame([mat2.agg(cv,axis=1),mat2.agg(np.nanmean,axis=1)],
                    index=['Coefficient of variation','Average of abundance '+text]).T
    df2['method']=labels[1]
    df=pd.concat([df1,df2],axis=0).reset_index(drop=True)


    ax = sns.jointplot(data=df, x='Average of abundance '+text, y='Coefficient of variation', 
                    hue="method", s=20,
                    alpha=1, palette=['#ffb300','#1e88e5'], legend=False)
    plt.legend(loc='upper left', labels=[
               labels[1],labels[0]], title=False, frameon=False)
    plt.savefig(name, dpi=400, bbox_inches='tight')


# violinplot
def violinplot(data,labels,anno=False):
    ax=plt.figure().add_subplot(111)
    violin=plt.violinplot(data,showextrema=False)
    for patch in violin['bodies']:
        patch.set_color('red')
        patch.set_facecolor('None')
        patch.set_alpha(0.7)
    plt.boxplot(data,showcaps=False,
                boxprops={'color':'#ff7f0e'},
                whiskerprops={'color':'#ff7f0e'},
                medianprops ={'linewidth':2},
                flierprops={'markeredgecolor':'#ff7f0e'})
    plt.xticks(list(range(1,len(labels)+1)),labels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if anno:
        mlist=[np.nanmedian(elem) for elem in data]
        print(mlist)
        mmax=max([max(elem) for elem in data])\
             +(max([max(elem) for elem in data])-min([min(elem) for elem in data]))*0.1
        for _i in range(len(data)):
            plt.text(_i+0.8,mmax,'median:'+str(mlist[_i]))
    

def splotFinAll2(_data, _data2, columns, path, title, max, min, color= ['#ffb300','#1e88e5']):
    plt.figure(figsize=(8,len(columns)*1.2))
    gs = gridspec.GridSpec(len(columns), 3, width_ratios=[1, 2, 1])
    for ii,scol in enumerate(columns):
        data=_data.loc[:,scol].reset_index().set_index('index').dropna(how='all')
        data.sort_values(by=scol, ascending=False, inplace=True)
        data2=_data2.loc[:,scol].reset_index().set_index('index').dropna(how='all')
        data2.sort_values(by=scol, ascending=False, inplace=True)
        overlap = set(data.index) & set(data2.index)

        d1 = pd.DataFrame(data.loc[~data.index.isin(overlap), scol])
        d2 = pd.DataFrame(data2.loc[~data2.index.isin(overlap), scol])
        O1 = pd.DataFrame(data.loc[data.index.isin(overlap), scol])
        O2 = pd.DataFrame(data2.loc[(O1.index), scol])
        
        ax1 = plt.subplot(gs[ii,0])
        ax1.scatter(range(d1.shape[0]), d1[scol].tolist(), s=2,
                    color=color[0],
                    alpha=0.6)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # ax1.set_ylabel('Protein Intensity')
        plt.ylim(min, max)
        if ii<len(columns)-1:
            ax1.set_xticks([])
        ax2 = plt.subplot(gs[ii,1])
        ax2.scatter(range(O1.shape[0]), O1[scol].tolist(), s=1,
                    color=color[0],
                    alpha=0.1)
        ax2.scatter(range(O2.shape[0]), O2[scol].tolist(), s=1,
                    color=color[1],
                    alpha=0.1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_yticks([])
        if ii<len(columns)-1:
            ax2.set_xticks([])
        plt.ylim(min, max)
        ax2.spines['right'].set_visible(False)
        # plt.text(O1.shape[0]/3, -1.2, 'Protein Rank')
        plt.text(0, max, scol,fontsize=8)
        ax3 = plt.subplot(gs[ii,2])
        ax3.scatter(range(d2.shape[0]), d2[scol].tolist(), s=2,
                    color=color[1],
                    alpha=0.6)
        ax3.spines['left'].set_visible(False)
        ax3.set_yticks([])
        plt.ylim(min, max)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        if ii<len(columns)-1:
            ax3.set_xticks([])
        plt.subplots_adjust(wspace=0.2)
    plt.savefig(path+'/'+title+'_.pdf', bbox_inches='tight',dpi=400)
        # return d1.shape[0], O1.shape[0], d2.shape[0]


def iqr(unlist):
    '''iqr四分位'''
    iqr = np.nanpercentile(unlist, 75) - np.nanpercentile(unlist, 25)
    maxfix = np.nanpercentile(unlist, 75) + 1.5*iqr
    minfix = np.nanpercentile(unlist, 25) - 1.5*iqr
    return maxfix, minfix


def moveouts(matrix, movemax=True, movemin=False):
    '''离群值处理'''
    unlist = matrix.unstack().tolist()
    maxfix, minfix = iqr(unlist)
    if movemax:
        matrix = matrix.applymap(
            lambda x: maxfix if x > maxfix else x)
    if movemin:
        matrix = matrix.applymap(
            lambda x: minfix if x < minfix else x)
    return matrix


def zscore(matrix):
    for i in range(len(matrix)):
        matrix.iloc[i, :] = sp.scale(matrix.iloc[i, :])
    return matrix

# try:
#     os.mkdir('output')
# except Exception:
#     print('The folder output already exists.')
# for mydir in ['2B','2D','2E']:
#     try:
#         os.mkdir('output/Figure'+mydir)
#     except Exception:
#         pass

#%% ##########Figure2
# Input Parameter
swlist=['FP','PD']
matlist=['abundance[log2]']

# Data Analysis
mat={}
for s in swlist:
    for m in matlist:
        mat['_'.join([s,m])]=pd.read_csv('_'.join([s,m])+'_tmt10.csv',index_col=0)
        # mat['_'.join([s,m])]=mat['_'.join([s,m])].apply(lambda x: x - np.mean(x), axis = 0)

# Figure2B --b.
for m in matlist:
    mat1=mat['_'.join([swlist[0],m])]
    mat2=mat['_'.join([swlist[1],m])]
    jointplot(mat1,mat2,name='FigureS2B_FPVSPD_'+m+'_jointplot_TMT10.pdf',
              labels=swlist,text='('+m+')')

    print("original matrix ks test:")
    print(ks_2samp(mat1.agg(cv,axis=1), mat2.agg(cv,axis=1)))
    print(ks_2samp(mat1.agg(np.nanmean,axis=1), mat2.agg(np.nanmean,axis=1)))

    # if m=='ratio':
    mat1=moveouts(mat1,movemin=True)
    mat2=moveouts(mat2,movemin=True)

    print("remove outlier matrix ks test:")
    print(ks_2samp(mat1.agg(cv,axis=1), mat2.agg(cv,axis=1)))
    print(ks_2samp(mat1.agg(np.nanmean,axis=1), mat2.agg(np.nanmean,axis=1)))

    jointplot(mat1,mat2,name='FigureS2B_FPVSPD_'+m+'_rm_outliers_jointplot_TMT10.pdf',
            labels=swlist,text='('+m+')')
    

# %%
