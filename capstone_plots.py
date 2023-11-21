
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def barplot(y_true, y_pred, file_name=None):
    """show in a paired barplot the amount of correctly predicted seizures and non-seizures

    Args:
        y_true: true labels
        y_pred: predicted labels
    """

    true_label = 'ground_truth'
    pred_label = 'correctly predicted'
    seizure_label = 'incoming seizure'
    non_seizure_label = 'no seizure incoming'

    # df = pd.DataFrame(columns=['amount', 'prediction', 'seizure'], dtype=np.int32)
    # true seizures
    df = pd.DataFrame()
    df.loc[1,'amount'] = y_true.count(1)
    df.loc[1,'prediction'] = true_label
    df.loc[1,'seizure'] = seizure_label
    df.loc[2,'amount'] = [not x for x in y_true].count(1)
    df.loc[2,'prediction'] = true_label
    df.loc[2,'seizure'] = non_seizure_label
    # correctly predicted seizures
    df.loc[3,'amount'] = [x & y for x,y in zip([x == 1 for x in y_pred], [x==1 for x in y_true])].count(1)
    df.loc[3,'prediction'] = pred_label
    df.loc[3,'seizure'] = seizure_label
    # correctly predicted non-seizures
    df.loc[4,'amount'] = [x & y for x,y in zip([x == 0 for x in y_pred], [x==0 for x in y_true])].count(1)
    df.loc[4,'prediction'] = pred_label
    df.loc[4,'seizure'] = non_seizure_label


    df['amount'] = df['amount'] / len(y_true) * 100
    df.head()

    rel_pred = list(df.loc[df['prediction'] != true_label, 'amount'].values / df.loc[df['prediction'] == true_label, 'amount'].values * 100)



    sns.set(style="whitegrid")

    #bar_color = '#b19ed5'
    bar_edge_color = '#977cca'
    background_color = "none"
    text_color = '#5a4275'
    custom_palette = ["#6A0572", "#AB83A1"]
    sns.set_palette(custom_palette)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(data=df, x='seizure', y='amount', hue='prediction',
                    palette=custom_palette,
                    edgecolor=bar_edge_color, 
                    # alpha=0.7, 
                    dodge=True, 
                    saturation=0.8)
    # for i, bars in enumerate(ax.containers):
    ax.bar_label(ax.containers[1], labels=[f"{x:.1f}%" for x in rel_pred],label_type='center', padding=3,
                    fontsize=18)
    legend = plt.legend(fontsize=24, loc='upper left', frameon=False, labelspacing=0.5, markerscale=1.5, prop={'weight': 'bold'})

    plt.xticks(fontsize=18, fontweight='bold', color=text_color)
    plt.xlabel(None, fontsize=18, fontweight='bold', color=text_color)
    plt.ylabel('percentage of samples', fontsize=18, fontweight='bold', color=text_color)
    # ax.set(ylabel = 'percentage of samples')
    # ax.set(xlabel = None)
    plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text

    plt.gca().set_facecolor('none')

    if file_name is not None:
        plt.savefig('Images/' + file_name, bbox_inches='tight', transparent=True)


def ana_barplot():
    sns.set(style="whitegrid")

    #bar_color = '#b19ed5'
    bar_edge_color = '#977cca'
    background_color = "none"
    text_color = '#5a4275'
    custom_palette = ["#6A0572", "#AB83A1"]
    #ustom_palette = ["#FF8C00", "#FFA07A"]  # Dark orange, Light orange
    #custom_palette = [(1.0, 0.4, 0.0, 1.0), (1.0, 0.8, 0.6, 1.0)]  
    sns.set_palette(custom_palette)

    # Create a bar plot for the average number of patients per age and gender with a transparent background
    plt.figure(figsize=(12, 6))
    bar_width = 0.7
    sns.barplot(x='Age Group', y='Number of Patients', hue='gender', data=df.groupby(['Age Group', 'gender']).size().reset_index().rename(columns={0: 'Number of Patients'}), 
                    ci=None, 
                    palette=custom_palette,
                    edgecolor=bar_edge_color, 
                    alpha=0.7, 
                    dodge=True, 
                    saturation=0.8)

    # Set labels and title
    plt.title('Average Number of Patients per Age Group and Gender', fontsize=20,fontweight='bold', color=text_color)
    plt.xlabel('Age Group', fontsize=18, fontweight='bold', color=text_color)
    plt.ylabel('Average Number of Patients', fontsize=18, fontweight='bold', color=text_color)

    # Set legend
    legend = plt.legend(title='Gender', title_fontsize='16', fontsize='14', loc='upper left', frameon=False, labelspacing=0.5, markerscale=1.5, prop={'weight': 'bold'})

    plt.xticks(fontsize=12, fontweight='bold', color=text_color)
    plt.yticks(fontsize=12, fontweight='bold', color=text_color)

    # Set the background color to be transparent
    plt.gca().set_facecolor('none')

    # Remove spines
    sns.despine()

    plt.savefig('genderage_plot.png', bbox_inches='tight', transparent=True)
    # Show the plot
    plt.show()
    


if __name__ == "__main__":
    # execute only if run as a script

    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
    

    barplot(y_true, y_pred)
    plt.show()


    # df.loc[1,'col2'] = 27
    # df.loc[1,'col3'] = 12
    # df.loc[1,'col4'] = 18

    print(df.head())
    a = 2
    

