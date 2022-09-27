#%%
from matplotlib import gridspec
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["pdf.fonttype"] = 42


ifile='FPVSPD_radar.csv'
data=pd.read_csv(ifile,index_col=0)
df=data.copy()
df.loc['Time2',:]=[1/(int(elem.split('min')[0]))\
                    for elem in df.loc['Time2',:]]

df.loc['NA ratio of quantified proteins',:]=[1/(float(elem.split('%')[0])+0.01)\
                                            # if float(elem.split('%')[0])!=0
                                            # else 3 for elem in df.loc['NA ratio of quantified proteins',:]]
                                            for elem in df.loc['NA ratio of quantified proteins',:]]
df.loc['Size of output files',:]=[1/float(elem.split('G')[0]) for elem in df.loc['Size of output files',:]]


#def min_max(dlist,m=0.4,M=1):
#    min_max_scaler = sp.MinMaxScaler(feature_range = (m,M),copy = False)
#    dlist=min_max_scaler.fit_transform(np.array(dlist).reshape(-1, 1))
#    return list(dlist.reshape(1, -1)[0])
#for i in range(1,len(df)):
#    df.iloc[i,:]=min_max(df.iloc[i,:])

for i in range(1,len(df)):
    df.iloc[i,:]=[float(x) for x in df.iloc[i,:]]
    df.iloc[i,:]=df.iloc[i,:]/max(df.iloc[i,:]) + 0.2
   
# tlist=['Time2','NA ratio of quantified proteins','# Quantified proteins3','# Quantified peptides3','Size of output files']
tlist = list(df.index)[1:]
plt.style.use('seaborn')
#fig=plt.figure(figsize=(5*1.5,4*1.5))
#gs=gridspec.GridSpec(1,3,fig)
for i,tmt in enumerate(list(set([elem.split('.')[0] for elem in df.columns]))):
    dff=df.loc[:,[elem for elem in df.columns if elem.startswith(tmt)]]
    dff.columns=dff.iloc[0,:]
    #dff=dff.T.loc[:,tlist].T
    pdlist=dff.iloc[1:,:].loc[:,'PD'].tolist()
    pdlist+=[pdlist[0]]
    fplist=dff.iloc[1:,:].loc[:,'FP'].tolist()
    fplist+=[fplist[0]]
    fig=plt.figure(figsize=(5*1.5,4*1.5))
    #ax=fig.add_subplot(gs[0,i], projection='polar')
    ax=fig.add_subplot(1,1,1, projection='polar')
    theta = list(np.arange(0.1,2,0.4)*np.pi)+[0.1*np.pi]


    plt.fill(np.linspace(0, 2*np.pi, 1000), np.ones(1000)*1.4, color='white', linestyle='-')
    for cycle in np.arange(0.2,1.3,0.2):
        #math.ceil(10*max(r))/10
        plt.plot(np.linspace(0, 2*np.pi, 1000), np.ones(1000)*cycle, color='gray', linestyle='-',alpha=0.5,lw=1,zorder=1)
    plt.ylim(0,1.3)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(tmt)
    plt.plot(theta,fplist,'o-',label='FP',color='#ffb300')
    plt.plot(theta,pdlist,'o-',label='PD',color='#1e88e5')
    for ii,vline in enumerate(theta[:-1]):
        plt.vlines(vline,0.2,1.2,color='gray',lw=1,alpha=0.5)
    plt.text(theta[0],1.3,tlist[0],fontsize=8)  
    plt.text(theta[1]+0.1*np.pi,1.3,tlist[1],fontsize=8)
    plt.text(theta[2],1.7,tlist[2],fontsize=8)
    plt.text(theta[3]-0.1*np.pi,1.8,tlist[3],fontsize=8)
    plt.text(theta[4],1.3,tlist[4],fontsize=8)

    colors=['#ffb300','#1e88e5']
    tlist2=['FP','PD']
    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(tlist2[i])) for i in range(len(colors))]
    ax.legend(handles=patches, bbox_to_anchor=(1,1), fontsize=10, frameon=False)
    
    plt.savefig('TMT-based_PDVSFP_radar_v2_'+tmt+'.pdf',bbox_inches='tight',dpi=400)

# %%
