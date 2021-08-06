import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.002, 336.304, 3509.327, 3521.313, 5052.072, 8099.117, 8105.111, 9499.974, 12281.015, 12284.003, 12284.255, 12299.582, 12299.666])
latency = np.array([12299.676, 336.298, 3173.019, 11.984, 1530.756, 3047.042, 5.991, 1394.860, 2781.037, 2.986, 0.250, 15.324, 0.081, 0.008])
event   = ["Interpreter", "CONV_2D", "CONV_2D", "MAX_POOL_2D", "CONV_2D", "CONV_2D", "MAX_POOL_2D", "CONV_2D", "CONV_2D", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX"]
colors = ["#1864ab", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9"]


ax1.barh(range(len(begin)),  latency, left=begin, color=colors)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (ms)")
plt.ylabel("Task")

data = [[ 0.002, 336.304, 3509.327, 3521.313, 5052.072, 8099.117, 8105.111, 9499.974, 12281.015, 12284.003, 12284.255, 12299.582, 12299.666],
        [ 336.298, 3173.019, 11.984, 1530.756, 3047.042, 5.991, 1394.860, 2781.037, 2.986, 0.250, 15.324, 0.081, 0.008],
        [ 117.584, 1020.239, 0.000, 488.897, 965.672, 0.000, 442.131, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]]

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