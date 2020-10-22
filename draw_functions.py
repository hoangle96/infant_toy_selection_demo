from matplotlib import pyplot as plt 
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np 

def draw_distribution(n_states, feature_vector, feature_names, pred, feature_labels):
    fig, axs = plt.subplots(nrows=n_states, ncols= len(feature_names),sharex='col', sharey='row',figsize=(14,14))
    state_color_dict = {0:'blue', 1:'green', 2:'purple', 3:'orange', 4:'yellow', 5:'chocolate', 6:'crimson', 7:'darkolivegreen'}
    for f_i in range(len(feature_vector.T)):
        feature = feature_vector.T[f_i]
        all_unique = np.unique(feature)
        x_labels = [str(int(x_i)) for x_i in all_unique]
        for idx, state in enumerate(range(n_states)):
            final_val = []
            final_height = []
            unique, cnt = np.unique(feature[pred==state], return_counts = True)
            height = cnt/cnt.sum()
            cnt_dict = {k: v for k,v in zip(unique, height)}

            for val in all_unique:
                final_val.append(val)
                if val in cnt_dict.keys():
                    final_height.append(cnt_dict[val])
                else:
                    final_height.append(0)
            axs[idx, f_i].set_ylim(top=1) 
            axs[idx, f_i].bar(final_val, final_height, color = state_color_dict[state])

            axs[idx, f_i].set_xticks(all_unique)
            axs[idx, f_i].set_xticklabels(labels = feature_labels[f_i], fontsize = 14)

            axs[idx, f_i].set_yticks(np.arange(0,1.1,0.5))
            axs[idx, f_i].set_yticklabels(labels = [str(np.around(y_i,1)) for y_i in np.arange(0,1.1,0.5)], fontsize = 14)
            axs[idx, 0].tick_params("y", left=False, labelleft=False)
            axs[idx, -1].tick_params("y", right=True, labelright=True)
            axs[0, f_i].title.set_text(feature_names[f_i])
            axs[idx, f_i].grid(False)
            secyax = axs[idx, 0].secondary_yaxis('left')
            secyax.set_ylabel("state "+str(state), fontsize = 14)
    return fig, axs

def draw_merged_timeline(k, merged_df_dict, state_list, time_list):
    df = merged_df_dict[k].copy()

    begin_time = df.iloc[0,:]
    begin_time = begin_time.loc['onset_mil']

    df = df.explode('toy')
    data1 = go.Scatter(
            x = df['onset_mil']/60000,
            y = df['toy'],
            mode='markers',
            marker=dict(color='rgba(131, 90, 241, 0)')
        )

    data2 = go.Scatter(
            x = df['offset_mil']/60000,
            y= df['toy'],
            mode='markers',
            marker=dict(color='rgba(131, 90, 241, 0)')
        )

    colors_dict = {'bucket':'orange', 'broom': 'blue', 'doll':'red', 'mom':'green', 'popper':'yellow', 'redball':'purple', 'stroller':'aqua', 'no_ops':'skyblue'}
    state_color_dict = {0:'blue', 1:'green', 2:'purple', 3:'red', 4:'yellow', 5:'chocolate', 6:'crimson', 7:'darkolivegreen'}
    
    # draw plain timeline 
    shapes=[dict(
            type = 'line',
            x0 = df['onset_mil'].iloc[i]/60000,
            y0 = df['toy'].iloc[i],
            x1 = df['offset_mil'].iloc[i]/60000,
            y1 = df['toy'].iloc[i],
            line = dict(
                color = colors_dict[df['toy'].iloc[i]],
                width = 2
            )    
        ) for i in range(len(df['onset_mil']))] 
    
    # draw the first patch
    shapes.extend([dict(type="rect",
                x0=(begin_time)/60000,
                y0=0,
                x1=time_list[0]/60000,
                y1=7,
                line=dict(
                    color="rgba(0, 0,0, 1)",
                    width=2,
                ),
               fillcolor=state_color_dict[state_list[0]],
               opacity = 0.05)])

    shapes.extend([dict(type="rect",
                x0=time_list[j-1]/60000,
                y0=0,
                x1=time_list[j]/60000,
                y1=7,
                line=dict(
                    color="rgba(0, 0, 0, 1)",
                    width=2,
                ),
                fillcolor=state_color_dict[state_list[j]],
                opacity = 0.05,
                       ) for j in range(len(time_list)) if j != 0])
    
    layout = go.Layout(
        shapes = shapes,
        title='Subject ' + str(k),
        height=400, width=900
    ) 

    # Plot the chart
    fig = go.Figure([data1, data2],layout)
    fig.update_layout(xaxis = dict(
            tickmode = 'linear',
            tick0 = 0.5,
            dtick = 0.5
        ), annotations=[
            dict(
                x=(time_list[m])/60000,
                y=6,
                text= str(state_list[m]),
            ) for m in range(len(time_list))])

    fig.update_layout(yaxis_type='category', showlegend=False)
    return fig