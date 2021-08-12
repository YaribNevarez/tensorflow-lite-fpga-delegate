import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.003, 0.021, 33.701, 33.773, 123.861, 135.871, 136.010, 180.257, 180.524, 254.855, 260.883, 261.412, 295.799, 296.847, 358.565, 361.552, 361.803, 377.096, 377.179])
latency = np.array([377.189, 33.696, 33.676, 90.158, 90.085, 12.007, 44.384, 44.244, 74.596, 74.329, 6.026, 34.914, 34.385, 62.765, 61.716, 2.984, 0.249, 15.289, 0.082, 0.007])
event   = ["Interpreter", "CONV_2D", "CONV_HW", "CONV_2D", "CONV_HW", "MAX_POOL_2D", "CONV_2D", "CONV_HW", "CONV_2D", "CONV_HW", "MAX_POOL_2D", "CONV_2D", "CONV_HW", "CONV_2D", "CONV_HW", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX"]
colors = ["#1864ab", "#4a98c9", "#94c4df", "#4a98c9", "#94c4df", "#4a98c9", "#4a98c9", "#94c4df", "#4a98c9", "#94c4df", "#4a98c9", "#4a98c9", "#94c4df", "#4a98c9", "#94c4df", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9"]


ax1.barh(range(len(begin)),  latency, left=begin, color=colors)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (ms)")
plt.ylabel("Task")

data = [[ 0.003, 33.701, 123.861, 135.871, 180.257, 254.855, 260.883, 295.799, 358.565, 361.552, 361.803, 377.096, 377.179],
        [ 33.696, 90.158, 12.007, 44.384, 74.596, 6.026, 34.914, 62.765, 2.984, 0.249, 15.289, 0.082, 0.007],
        [ 33.676, 90.085, 0.000, 44.244, 74.329, 0.000, 34.385, 61.716, 0.000, 0.000, 0.000, 0.000, 0.000]]

columns = ("CONV_2D", "CONV_2D", "MAX_POOL_2D", "CONV_2D", "CONV_2D", "MAX_POOL_2D", "CONV_2D", "CONV_2D", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX")
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