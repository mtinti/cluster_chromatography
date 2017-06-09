# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as sch
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
norMax = lambda x: x / x.max()

           
def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)        
        
        
def main(in_df, cut_distance_cols='', cut_distance_rows='', 
         method='ward', metric='euclidean',
         cluster_columns = True,
         order_row_and_columns=True,
         interpolation=True, figsize=None, 
         color_map_id='Blues',
         step_first_x=5,
         color_bar = True,
         col_to_mw = '' ,
         xTitle_padding=1.05,
         title='Main Title',
         add_second_axis={}):

    if figsize == None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)        

    
    gs = gridspec.GridSpec(2, 2,
                        width_ratios=[1, 4],
                        height_ratios=[1, 4],
                        wspace=0.05,hspace=0.05
                       )

    #corner top left, placeholder for colormap
    ax1 = plt.subplot(gs[0])
    clean_axis(ax1)
    
    #the dendogram for the columns
    ax2 = plt.subplot(gs[1])
    if cluster_columns == True:
        #print in_df.T.head()
        link = sch.linkage(in_df.T, method, metric)
        den_cols = sch.dendrogram(link, color_threshold=cut_distance_cols, ax=ax2, orientation='top')
        ax2.set_ylabel('Distance')
        ax2.yaxis.set_label_position('right') 
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_position(('outward', 10)) 
        #clean_axis(ax2)
        ax2.tick_params(axis='y',# changes apply to the x-axis
                    which='both',# both major and minor ticks are affected
                    left='off', 
                    right='on',
                    labelright ='on',
                    labelleft='off')        
        ax2.tick_params(axis='x',# changes apply to the x-axis
                    which='both',# both major and minor ticks are affected
                    bottom='off', 
                    top='off',
                    labelbottom='off',
                    labeltop='off') 
        fig.suptitle(title, fontsize=16, y = 0.95)
    else:
        clean_axis(ax2)
        fig.suptitle(title, fontsize=16, y = 0.8)
        
    
    
    
    #the dendogram for the rows
    ax3 = plt.subplot(gs[2])
    #print in_df.head()
    link = sch.linkage(in_df, method, metric)
    den_rows = sch.dendrogram(link, color_threshold=cut_distance_rows, ax=ax3, orientation='left')
    ax3.set_xlabel('Distance')
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)  
    ax3.spines['right'].set_visible(False)     
    ax3.spines['bottom'].set_visible(True)
    ax3.spines['bottom'].set_position(('outward', 10)) 
    
    ax3.tick_params(axis='x',# changes apply to the x-axis
                    which='both',# both major and minor ticks are affected
                    bottom='on', 
                    top='off',
                    labelbottom='on',
                    labeltop='off')
    
    ax3.tick_params(axis='y',# changes apply to the x-axis
                    which='both',# both major and minor ticks are affected
                    left='off', 
                    right='off',
                    labelright ='off',
                    labelleft='off')  
    

 
    #the heatmap plot
    ax4 = plt.subplot(gs[3])
    
    if len(add_second_axis) > 0:
        #add new axes
        new_ax = ax4.twiny()
        new_ax.plot([0,in_df.shape[1]],[0,in_df.shape[1]],c='w',alpha=0.01)
        new_ax.tick_params(axis='x',# changes apply to the x-axis
                    which='both',# both major and minor ticks are affected
                    bottom='on', 
                    top='off',
                    labelbottom='on',
                    labeltop='off',
                    length = 5)
        new_ax.set_ylim(0,in_df.shape[0])
        new_ax.set_xlim(0,in_df.shape[1])
        new_ax.spines['top'].set_visible(False)
        new_ax.spines['left'].set_visible(False)  
        new_ax.spines['right'].set_visible(False)  
        new_ax.spines['bottom'].set_position(('outward', 50))
        new_ax.set_xlabel(add_second_axis['label'])
        new_ax.xaxis.set_label_coords(1.05, -0.12)
        xticks = [n-0.5 for n in add_second_axis['values'].keys()]
        new_ax.set_xticks(xticks)
        xtickslabels = [add_second_axis['values'][n+0.5] for n in xticks]
        new_ax.set_xticklabels(xtickslabels)
    if order_row_and_columns == True:
        heatmap=ax4.pcolor(in_df.iloc[den_rows['leaves'],den_cols['leaves']])
    else:
        heatmap=ax4.pcolor(in_df.ix[den_rows['leaves']])
    ax4.set_aspect('auto')
    ax4.set_ylim(0,in_df.shape[0])
    ax4.set_xlim(0,in_df.shape[1])
    heatmap.set_cmap(color_map_id)
    
    ax4.tick_params(axis='x',# changes apply to the x-axis
                    which='both',# both major and minor ticks are affected
                    bottom='on', 
                    top='off',
                    labelbottom='on',
                    labeltop='off')
    
    ax4.tick_params(axis='y',# changes apply to the x-axis
                    which='both',# both major and minor ticks are affected
                    left='off', 
                    right='off',
                    labelright ='off',
                    labelleft='off') 
         
    ax4.spines['top'].set_visible(False)
    ax4.spines['left'].set_visible(False)  
    ax4.spines['right'].set_visible(False)  
    ax4.spines['bottom'].set_position(('outward', 10)) 
    ax4.set_xlabel('Fraction')
    ax4.xaxis.set_label_coords(1.05, -0.03)
    
    if step_first_x == 1:
        xticks = np.arange(0.5, in_df.shape[1], step_first_x)
        ax4.set_xticks(xticks)
        ax4.set_xticklabels([str(int(n+0.5)) for n in xticks])
    else:
        tick_range = np.arange(step_first_x-0.5, in_df.shape[1], step_first_x)
        xticks = [0.5]+[n for n in tick_range]
        ax4.set_xticks(xticks)
        ax4.set_xticklabels([str(int(n+0.5)) for n in xticks])

    cbaxes = fig.add_axes([1, 0.4, 0.03, 0.2])    
    plt.colorbar(heatmap, cax = cbaxes, orientation='vertical')
    cbaxes.spines['right'].set_position(('outward', 5))
    fig.savefig(title+'.svg')
    plt.show()
    



if __name__ == '__main__':
    
    in_file = 'test_data/LFQ-intensity_peaks.txt'
    test_df = pd.DataFrame.from_csv(in_file, sep='\t')
    test_df = test_df.apply(norMax,1)
    test_df = test_df.dropna()
    test_df[test_df<0.01]=0
    test_df = test_df.sample(50)
    #test_df = test_df.iloc[0:20,:]
    #g = sns.clustermap(test_df,col_cluster=False )
    #plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    main(
         test_df,
         method='ward', 
         metric='euclidean',
         cluster_columns = True,
         order_row_and_columns= True,
         color_map_id='Blues',
         figsize=(12,12),
         cut_distance_cols=25,
         cut_distance_rows=20,
         step_first_x = 5,
         title='Main Title',
         add_second_axis={'label':'MW kDa','values': {
                                                                      
                          15:'1.300',
                          22:'660', 
                          34:'150', 
                          37:'66.4', 
                          41:'17', 
                          47:'0.1'} 
                          }
         )
    
    


