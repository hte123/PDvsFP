#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tmtc.py
@Time    :   2021/11/24 15:43:49
@Author  :   Youqi Liu
@Version :   1
@Contact :   liuyouqi@westlakeomics.com

2022.02.19 edited by Tianen He
1. change colors of Figure2B,2E
2. change Figure2F to Spearman's correlation 

2022.03.08 edited by T He
add Kolmogorov-Smirnov test to Figure2B

2022.03.11 edited by T He
1. change texts of legends, labels, titles
2. change plt.rcParams
3. change legend position of Figure3A

2022.04.07 edited by T He
1. edit remove outliers to Figure2B

2022.09.17 edited by T He
rename output files

2022.09.25 T He
replace -Inf with NA in log2 abundance matrix for Figure2F
'''

#%% ##########Function definition and import packages
import pandas as pd
import os
from scipy.stats import ks_2samp

# from seaborn.relational import scatterplot
os.environ['R_HOME'] = 'D:/Program Files/R-4.2.1' #set your R path
# from rpy2.robjects.packages import importr
# from rpy2.robjects.vectors import FloatVector
from rpy2 import robjects
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
# import umap
import umap.umap_ as umap
from sklearn import preprocessing as sp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def to_percent(temp,position):
    return '%1.0f' % (100*temp) + '%'

# rep correlation
robjects.r('''
            ge.plot.techrep.correlation <- function(cor1,cor2,xlab = "repA",ylab = "repB",name="pearson_correlation"){
            pdf(paste0(name,".pdf"))
            res <- cor.test(cor1, cor2, 
            use = "pairwise.complete.obs",
                            alternative = "two.sided",
                            method = "spearman")
            r <- res$estimate
            # p <- res$p.value
            # print(p)

            cor1[sapply(cor1,is.infinite)] = NA
            cor2[sapply(cor2,is.infinite)] = NA

            smoothScatter(cor1, cor2, nrpoints = 100,cex = 2,
                            colramp = colorRampPalette(c(blues9,"orange", "red")),
                            main = name, xlab = xlab, ylab = ylab)
            abline(lm(cor1 ~ cor2), col="red", lwd=2, lty=2)
            text(min(cor1,na.rm = T)*1.8,max(cor2,na.rm = T)*0.8,
                 labels =paste0( "r =", as.character(round(r,4))),cex = 1.2, adj=c(0, 0))
            # text(min(cor1,na.rm = T)*1.8,max(cor2,na.rm = T)*0.75,
            #      labels =paste0( "p-value =", as.character(p)),cex = 1.2)
            dev.off()
            }
           ''')

def plot_cor(cor1,cor2,xlab = "repA",ylab = "repB",name="pearson_correlation"):
    robjects.r['ge.plot.techrep.correlation'](robjects.FloatVector(cor1),
                                            robjects.FloatVector(cor2),
                                            robjects.FactorVector([xlab]),
                                            robjects.FactorVector([ylab]),
                                            robjects.FactorVector([name]))
    print('ge.plot.techrep.correlation succeed:',name+'.pdf')

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
    

def splotFinAll2(_data, _data2, columns,path, title, m=7,color= ['#ffb300','#1e88e5']):
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
        plt.ylim(-0.2, m)
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
        plt.ylim(-0.2, m)
        ax2.spines['right'].set_visible(False)
        # plt.text(O1.shape[0]/3, -1.2, 'Protein Rank')
        plt.text(0, m, scol,fontsize=8)
        ax3 = plt.subplot(gs[ii,2])
        ax3.scatter(range(d2.shape[0]), d2[scol].tolist(), s=2,
                    color=color[1],
                    alpha=0.6)
        ax3.spines['left'].set_visible(False)
        ax3.set_yticks([])
        plt.ylim(-0.2, m)
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
    print("maxfix = " + str(maxfix) + ". minfix = " + str(minfix))
    if movemax:
        matrix = matrix.applymap(
            lambda x: maxfix if x > maxfix else x)
    if movemin:
        matrix = matrix.applymap(
            lambda x: minfix if x < minfix else x)
    return matrix

def barplot(data,x_data,colors):
    f, ax = plt.subplots(figsize=(4, 6))
    x = data.columns
    for i, col in enumerate(data.index):
        height=pvca.iloc[i,:]
        if i == 0:
            plt.bar(x=x_data,height=height,width=0.145,
                    color=colors[i],label=data.index[i])
            bottom = np.array(height)
        else:
            plt.bar(x=x_data,height=height,bottom=bottom,width=0.145,
                    color=colors[i],label=data.index[i])
            bottom = np.array(bottom)+np.array(height)
    plt.xticks(x_data,x,rotation=45)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.85),
              frameon=False)


def zscore(matrix):
    for i in range(len(matrix)):
        matrix.iloc[i, :] = sp.scale(matrix.iloc[i, :])
    return matrix

def ucdata(_matrix, method='pca',rs=1):
    # log2 matrix/ratio original matrix
    # _matrix = matrix.fillna(0)
    # _matrix = zscore(_matrix)
    features = _matrix.shape[0]
    if method == 'pca':
        pca_n = PCA(n_components=2, random_state=rs)
        pca_df = pca_n.fit_transform(_matrix.T)
        pc1, pc2 = pca_n.explained_variance_ratio_
        pca = pd.DataFrame(pca_df, columns=['X1', 'X2'])
        data = pca
    elif method == 'tsne':
        tsne = TSNE(random_state=rs)
        tsne.fit_transform(_matrix.T)
        tsne = pd.DataFrame(tsne.embedding_,
                            index=_matrix.columns)
        tsne.columns = ['X1', 'X2']
        data = tsne
    elif method == 'umap':
        embedding = umap.UMAP(n_neighbors=15,n_epochs=200,min_dist=0.1,
                              init='spectral',learning_rate=1,
                              random_state=rs).fit_transform(_matrix.T)
        Myumap = pd.DataFrame(embedding,
                              index=_matrix.columns)
        Myumap.columns = ['X1', 'X2']
        data = Myumap
    else:
        print('This method is not currently supported')
    data.index = _matrix.columns
    return data

# try:
#     os.mkdir('output')
# except Exception:
#     print('The folder output already exists.')
# for mydir in ['2B','2D','2E','2F','3A','3B']:
#     try:
#         os.mkdir('output/Figure'+mydir)
#     except Exception:
#         pass

#%% ##########Figure2
# Input Parameter
swlist=['FP','PD']
matlist=['abundance[log2]','ratio']
pairs={
    'control cortex':['b4_133C','b2_133N'],
    'covid medulla':['b2_133C','b1_129C'],
    'covid cortex':['b1_133C','b3_128N']
    }

# Data Analysis
mat={}
for s in swlist:
    for m in matlist:
        mat['_'.join([s,m])]=pd.read_csv('_'.join([s,m])+'_tmt16.csv',index_col=0)
        # mat['_'.join([s,m])]=mat['_'.join([s,m])].apply(lambda x: x - np.mean(x), axis = 0)

#%%
# Figure2B --b.
for m in matlist:
    mat1=mat['_'.join([swlist[0],m])]
    mat2=mat['_'.join([swlist[1],m])]
    jointplot(mat1,mat2,name='Figure2B_FPVSPD_'+m+'_jointplot_tmt16.pdf',
              labels=swlist,text='('+m+')')
    
    print("original matrix ks test:")
    print(ks_2samp(mat1.agg(cv,axis=1), mat2.agg(cv,axis=1)))
    print(ks_2samp(mat1.agg(np.nanmean,axis=1), mat2.agg(np.nanmean,axis=1)))

    if m=='ratio':
        mat1=moveouts(mat1,movemin=True)
        mat2=moveouts(mat2,movemin=True)

        print("remove outlier matrix ks test:")
        print(ks_2samp(mat1.agg(cv,axis=1), mat2.agg(cv,axis=1)))
        print(ks_2samp(mat1.agg(np.nanmean,axis=1), mat2.agg(np.nanmean,axis=1)))

        jointplot(mat1,mat2,name='Figure2B_FPVSPD_'+m+'_rm_outliers_jointplot_tmt16.pdf',
                labels=swlist,text='('+m+')')
    
#%%
# Figure2D --d.	Correlation coefficients between 
# abundance ratios of the same sample quantified by PD and FP
# for m in matlist:
#     mat1=mat['_'.join([swlist[0],m])]
#     mat2=mat['_'.join([swlist[1],m])]
#     corrlist=[]
#     for i in range(len(mat1.columns)):
#         tmp=pd.merge(mat1.iloc[:,i],mat2.iloc[:,i],
#                      left_index=True,right_index=True).dropna(how='any')
#         corrlist.append(tmp.corr().iloc[0,1])
#     violinplot([corrlist],labels=[m],anno=True)
#     plt.savefig('output/Figure2D/Figure2D_FPVSPD_'+m+'_boxplot.pdf')

#     if m=='ratio':
#         mat1=moveouts(mat1,movemin=True)
#         mat2=moveouts(mat2,movemin=True)
#         corrlist=[]
#         for i in range(len(mat1.columns)):
#             tmp=pd.merge(mat1.iloc[:,i],mat2.iloc[:,i],
#                         left_index=True,right_index=True).dropna(how='any')
#             corrlist.append(tmp.corr().iloc[0,1])
#         violinplot([corrlist],labels=[m],anno=True)
#         plt.savefig('output/Figure2D/Figure2D_FPVSPD_'+m+'_rm_outliers_boxplot.pdf')


# Figure 2E --e. 
cols=[]
for elem in pairs.values():
    cols+=elem
for m in matlist:
    mat1=mat['_'.join([swlist[0],m])]
    mat2=mat['_'.join([swlist[1],m])]
    mm = int(max(mat1.max().max(), mat2.max().max()))+1
    splotFinAll2(mat1, mat2,columns=cols,path='./', title='Figure2E_PD_Fragpipe_rank_by_PD_'+m+'_tmt16', m=mm)
    if m=='ratio':
        mat1=moveouts(mat1,movemin=True)
        mat2=moveouts(mat2,movemin=True)
        mm = int(max(mat1.max().max(), mat2.max().max()))+1
        splotFinAll2(mat1, mat2,columns=cols,path='./', title='Figure2E_PD_Fragpipe_rank_by_PD_'+m+'_rm_outliers_tmt16', m=mm)

#%%
# Figure 2F --technical replicate 
# b4_133C b2_133N control cortex 
# b2_133C b1_129C covid medulla
# b1_133C b3_128N covid cortex
for s in swlist:
    for m in matlist:
        data=mat['_'.join([s,m])].copy()
        for key,values in pairs.items():
            tmpdf=data.loc[:,values].dropna(how='all')
            plot_cor(tmpdf[values[0]].tolist(),tmpdf[values[1]].tolist(),
                     xlab=values[0],ylab=values[1],
                     name='Figure2F_'+'_'.join([s,m,key])+'_spearman_correlation_tmt16')
        # if m=='ratio':
        #     data=moveouts(data,movemin=True)
        #     for key,values in pairs.items():
        #         tmpdf=data.loc[:,values].dropna(how='all')
        #         plot_cor(tmpdf[values[0]].tolist(),tmpdf[values[1]].tolist(),
        #                 xlab=values[0],ylab=values[1],
        #                 name='Figure2F_'+'_'.join([s,m,key]+'_tmt16')+'_rm_outliers_spearman_correlation')
            
# Figure2G
# Output File

# %% ##########Figure3
# Input Parameter
pvca_file='pvca_summary.tsv'

swlist=['FP','PD']
matlist=['batch','batchfree_norm']
info_file='cvdnx_kidney_label20210801_forBatchServer_b1-4.csv'

# Data Analysis

# Figure 3A 
pvca=pd.read_csv(pvca_file,sep='\t',index_col=0).T
pvca=pvca.applymap(lambda x:float(x.strip('%'))/100)

colors=['#fae43c','#43284e','#4c458a','#3a6d90','#31938f','#90c959','#39ba83']
x_data=[0.9,1.3,1.05,1.45]
barplot(pvca,x_data,colors)
plt.legend(bbox_to_anchor = (-0.15, -0.15), loc = "upper left", ncol = 2)
plt.savefig('Figure3A_PVCA_tmt16.pdf',bbox_inches='tight',dpi=400)


#%%
# Figure 3B
info=pd.read_csv(info_file)
colors=['#f37673','#24bec0','#9a81bc','#7dae41']
markdict={'b1':'o','b2':'^','b3':'s','b4':'p'}
colordict=dict(zip(['Cortex COVID', 'Cortex non-COVID', 
                    'Medulla COVID', 'Medulla non-COVID'],
                    colors))

mat={}
for s in swlist:
    for m in matlist:
        data=pd.read_csv('_ratio_53_limma_'.join([s,m])+'.csv',
                         index_col=0).loc[:,info['MS_Sample ID']]
        mat['_ratio_53_limma_'.join([s,m])]=data.copy()
        for method in ['umap']:
            df=ucdata(data,method=method,rs=2)

            for type in info.columns[1:]:
                cdict={}
                label=info[type].tolist()
                for _i,elem in enumerate(info[type].unique()):
                    cdict[elem]=colors[_i]
                clist=[cdict.get(el) for el in label]
                patches = [mpatches.Patch(color=values, label="{:s}".format(key)) 
                           for key,values in cdict.items()]
                plt.figure(figsize=(6,5))
                plt.xlabel('')
                plt.ylabel('')
                plt.title('_'.join([s,m,method,type]))
                plt.scatter(x=df['X1'],y=df['X2'],c=clist)
                plt.legend(handles=patches,frameon=False)
                plt.savefig('Figure3B_'+'_'.join([s,m,method,type])+'_tmt16.pdf',
                            dpi=400,bbox_inches='tight')
            fig=plt.figure(figsize=(9,5))
            gs=gridspec.GridSpec(1,20,figure=fig)
            dff=pd.merge(df,info.set_index('MS_Sample ID'),
                         left_index=True,right_index=True)
            ax=fig.add_subplot(gs[0,:14])
            plt.xlabel('')
            plt.ylabel('')
            plt.title('_'.join([s,m,method]))
            for j in range(len(dff)):
                dlist=dff.iloc[j,:].tolist()
                if dlist[2]=='COVID':
                    dlist[3]=' '.join([dlist[3],dlist[2]])
                else:
                    dlist[3]=' '.join([dlist[3],'non-COVID'])
                
                plt.scatter(dlist[0],dlist[1],
                            color=colordict[dlist[3]],
                            marker=markdict[dlist[4]])

            patches1 = [mpatches.Patch(color=values, label="{:s}".format(key)) 
                           for key,values in colordict.items()]
            ml=plt.legend(handles=patches1,frameon=False,bbox_to_anchor=(1.4,1))

            ax2=fig.add_subplot(gs[0,19:])
            for key,values in markdict.items():
                plt.scatter(0,0,marker=values,
                            label=key,color='black')
                
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            plt.xticks([])
            plt.yticks([])
            plt.legend(frameon=False,bbox_to_anchor=(-3.4,0.75))
            plt.scatter(0,0,color='white',marker='s')
            plt.savefig('Figure3B_'+'_'.join([s,m,method])+'_tmt16.pdf',
                            dpi=400,bbox_inches='tight')
# Output File

# %%
