import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

#Python:
begin   = np.array([0.000, 0.004, 0.010, 0.029, 0.039, 0.055, 20.006, 35.329, 35.333, 35.489, 35.543, 35.698, 50.674, 55.450, 55.453, 55.685, 55.709, 55.942, 60.565, 61.999, 62.002, 62.547, 62.559, 63.104, 65.023, 65.265, 79.050, 79.131, ])
latency = np.array([79.141, 20.000, 0.029, 19.897, 0.025, 19.711, 15.320, 15.342, 0.210, 15.156, 0.206, 14.553, 4.774, 5.114, 0.256, 4.868, 0.250, 4.467, 1.432, 3.022, 0.556, 2.323, 0.552, 1.911, 0.240, 13.781, 0.079, 0.008, ])
event   = ["MODEL", "CONV_2D", "DELEGATE", "HARDWARE", "DELEGATE", "HARDWARE", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "DELEGATE", "HARDWARE", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "DELEGATE", "HARDWARE", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "DELEGATE", "HARDWARE", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX", ]
colors  = ["#1864ab", "#4a98c9", "#6faed4", "#94c4df", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", ]

data = [[0.004, 20.006, 35.329, 50.674, 55.450, 60.565, 61.999, 65.023, 65.265, 79.050, 79.131, ],
        [ 20.000, 15.320, 15.342, 4.774, 5.114, 1.432, 3.022, 0.240, 13.781, 0.079, 0.008, ],
        [ 19.897, 0.000, 15.156, 0.000, 4.868, 0.000, 2.323, 0.000, 0.000, 0.000, 0.000, ]]
columns = ("CONV_2D", "MAX_POOL_2D", "CONV_2D", "MAX_POOL_2D", "CONV_2D", "MAX_POOL_2D", "CONV_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX", )


ax1.barh(range(len(begin)),  latency, left=begin, color=colors)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (ms)")
plt.ylabel("Task")

rows = ["Hardware", "Software", "II OFFSET"]

# Get some pastel shades for the colors
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    ax2.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(data[row])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

plt.sca(ax2)
# Add a table at the bottom of the axes
the_table = ax2.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom',
                      fontsize='xx-small')

the_table.auto_set_font_size(False)
the_table.set_fontsize(7)


# Adjust layout to make room for the table:

plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Latency (ms)")

plt.xticks([])
ax2.grid(linestyle = ':')


plt.show()