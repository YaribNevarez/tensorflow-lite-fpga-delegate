import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

#Python:
begin   = np.array([0.000, 0.004, 0.010, 0.029, 40.167, 63.559, 86.940, 102.263, 102.268, 102.456, 138.872, 147.633, 156.386, 162.129, 162.132, 162.680, 184.800, 189.178, 193.548, 196.388, 196.630, 210.419, 210.501, ])
latency = np.array([210.510, 40.160, 0.034, 39.982, 23.388, 23.378, 15.320, 36.607, 0.295, 36.352, 8.759, 8.750, 5.740, 22.670, 0.600, 22.087, 4.375, 4.368, 2.838, 0.239, 13.785, 0.080, 0.008, ])
event   = ["MODEL", "CONV_2D", "DELEGATE", "HARDWARE", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "MUL", "ADD", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX", ]
colors  = ["#1864ab", "#4a98c9", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", ]

data = [[0.004, 424.412, 447.801, 471.181, 486.547, 2322.448, 2331.207, 2339.959, 2345.686, 3603.309, 3607.687, 3612.057, 3614.899, 3615.141, 3628.933, 3629.015, ],
        [ 424.404, 23.385, 23.377, 15.363, 1835.898, 8.757, 8.749, 5.725, 1257.620, 4.375, 4.369, 2.839, 0.239, 13.788, 0.080, 0.008, ],
        [ 39.982, 0.000, 0.000, 0.000, 36.352, 0.000, 0.000, 0.000, 22.087, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, ]]
columns = ("CONV_2D", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "MUL", "ADD", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX", )



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