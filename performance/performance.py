import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

#Python:
begin   = np.array([0.000, 0.002, 0.071, 0.098, 0.108, 0.131, 19.204, 42.597, 65.979, 81.375, 82.212, 82.503, 82.546, 82.858, 100.889, 109.653, 118.405, 124.117, 126.556, 127.216, 127.232, 128.069, 138.671, 143.052, 147.423, 150.288, 150.530, 151.691, ])
latency = np.array([151.702, 19.198, 0.036, 19.025, 0.031, 18.914, 23.390, 23.379, 15.393, 19.510, 0.332, 18.351, 0.350, 17.660, 8.760, 8.750, 5.710, 14.550, 0.674, 11.433, 0.849, 10.462, 4.378, 4.369, 2.863, 0.239, 1.159, 0.009, ])
event   = ["MODEL", "CONV_2D", "DELEGATE", "HARDWARE", "DELEGATE", "HARDWARE", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "DELEGATE", "HARDWARE", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "DELEGATE", "HARDWARE", "DELEGATE", "HARDWARE", "MUL", "ADD", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX", ]
colors  = ["#1864ab", "#4a98c9", "#6faed4", "#94c4df", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#6faed4", "#94c4df", "#6faed4", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", ]

data = [[0.004, 423.687, 447.075, 470.455, 485.815, 2322.040, 2330.799, 2339.551, 2345.252, 3603.124, 3607.501, 3611.872, 3614.706, 3614.947, 3616.109, ],
        [ 423.679, 23.385, 23.378, 15.358, 1836.222, 8.756, 8.750, 5.699, 1257.869, 4.375, 4.368, 2.832, 0.239, 1.159, 0.008, ],
        [ 19.025, 0.000, 0.000, 0.000, 18.351, 0.000, 0.000, 0.000, 11.433, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, ]]
columns = ("CONV_2D", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "MUL", "ADD", "MAX_POOL_2D", "CONV_2D", "MUL", "ADD", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX", )

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