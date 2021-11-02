import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

#Python:
begin   = np.array([0.000, 0.004, 0.008, 0.026, 33.111, 45.527, 45.531, 45.691, 79.291, 85.408, 85.411, 86.032, 110.707, 113.741, 113.999, 125.755, 125.821, ])
latency = np.array([125.831, 33.105, 0.033, 32.960, 12.413, 33.761, 0.241, 33.535, 6.115, 25.297, 0.666, 24.639, 3.032, 0.255, 11.753, 0.065, 0.008, ])
event   = ["MODEL", "CONV_2D", "DELEGATE", "HARDWARE", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX", ]
colors  = ["#1864ab", "#4a98c9", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", ]


data = [[0.003, 340.883, 353.153, 1923.492, 1929.562, 3360.153, 3363.175, 3363.433, 3375.196, 3375.263, ],
        [ 340.876, 12.268, 1570.337, 6.068, 1430.588, 3.020, 0.255, 11.760, 0.065, 0.008, ],
        [ 32.960, 0.000, 33.535, 0.000, 24.639, 0.000, 0.000, 0.000, 0.000, 0.000, ]]
columns = ("CONV_2D", "MAX_POOL_2D", "CONV_2D", "MAX_POOL_2D", "CONV_2D", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX", )

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