import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import preprocess


def split_data_by_label(data: dict):
    data_0 = {}
    data_1 = {}
    for patient_num, patient_data in data.items():

        if 1 in patient_data['SepsisLabel'].values:
            data_1[patient_num] = patient_data
        else:
            data_0[patient_num] = patient_data

    return data_0, data_1


if __name__ == '__main__':
    path_to_train = 'data/train/'
    train = preprocess.load_data_to_dict(path_to_train)

    train_0, train_1 = split_data_by_label(train)
    min_0 = 500
    min_1 = 500
    max_0 = 0
    max_1 = 0
    sum_0 = 0
    sum_1 = 0
    for patient_num, patient_data in train_0.items():
        num_hours = patient_data.shape[0]
        sum_0 += num_hours
        if num_hours < min_0:
            min_0 = num_hours
        if num_hours > max_0:
            max_0 = num_hours

    print('Negative Sepsis:')
    print(f'avg num hours: {sum_0 / len(train_0.items())}, min num of hours: {min_0}, max num of hours: {max_0}')

    for patient_num, patient_data in train_1.items():
        num_hours = patient_data.shape[0]
        sum_1 += num_hours
        if num_hours < min_1:
            min_1 = num_hours
        if num_hours > max_1:
            max_1 = num_hours

    print('Positive Sepsis:')
    print(f'avg num hours: {sum_1 / len(train_1.items())}, min num of hours: {min_1}, max num of hours: {max_1}')

    avg_hours_0 = sum_0 / len(train_0.items())
    avg_hours_1 = sum_1 / len(train_1.items())

    # Data to plot
    labels = ['Avg num hours']
    negative_sepsis = [avg_hours_0]
    positive_sepsis = [avg_hours_1]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, negative_sepsis, width, label='Negative Sepsis')
    rects2 = ax.bar(x + width / 2, positive_sepsis, width, label='Positive Sepsis')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Hours')
    ax.set_title('Comparison of Average Hours in Negative and Positive Sepsis')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Function to auto-label the bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

    sepsis_positive_data = pd.concat(train_1.values())

    mean_sepsis_pos = sepsis_positive_data.mean()
    var_sepsis_pos = sepsis_positive_data.var()

    sepsis_negative_data = pd.concat(train_0.values())
    # Create an empty list to store the results
    table_data = []

    # Calculate the statistics and store them in the list
    for col in sepsis_negative_data.columns:
        t_statistic, p_value = ttest_ind(sepsis_negative_data[col], sepsis_positive_data[col], nan_policy="omit")
        mean_1 = sepsis_negative_data[col].mean()
        mean_2 = sepsis_positive_data[col].mean()

        table_data.append([col, f"{mean_1:.2f}", f"{mean_2:.2f}", f"{t_statistic:.2f}", f"{p_value:.2f}"])

    # Create a DataFrame from the list
    table_columns = ['Column', 'Mean 1 (Negative Sepsis)', 'Mean 2 (Positive Sepsis)', 't-statistic', 'p-value']
    stats_table = pd.DataFrame(table_data, columns=table_columns)

    # Set up the figure and axis for the table
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.4))

    # Hide the axis
    ax.axis('off')

    # Create the table and add it to the axis
    table_plot = ax.table(cellText=stats_table.values,
                          colLabels=stats_table.columns,
                          cellLoc='center',
                          loc='center')

    # Set the font size and auto-scale columns
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.auto_set_column_width(col=list(range(len(stats_table.columns))))

    # Save the table as an image
    plt.savefig('table_plot.png', bbox_inches='tight')

    # Show the table plot (optional)
    plt.show()
