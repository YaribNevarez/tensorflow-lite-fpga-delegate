import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.002, 210.872, 2254.418, 2261.415, 3240.497, 5188.016, 5191.525, 6083.184, 7861.031, 7862.783, 7862.945, 7874.015, 7874.079])
latency = np.array([7874.091, 210.866, 2043.545, 6.995, 979.081, 1947.517, 3.508, 891.656, 1777.844, 1.749, 0.159, 11.067, 0.061, 0.009])
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

data = [[ 0.002, 210.872, 2254.418, 2261.415, 3240.497, 5188.016, 5191.525, 6083.184, 7861.031, 7862.783, 7862.945, 7874.015, 7874.079],
        [ 210.866, 2043.545, 6.995, 979.081, 1947.517, 3.508, 891.656, 1777.844, 1.749, 0.159, 11.067, 0.061, 0.009],
        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]]

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