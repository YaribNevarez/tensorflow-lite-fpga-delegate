import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.004, 210.877, 2254.415, 2261.413, 3240.436, 5187.931, 5191.439, 6083.089, 7860.954, 7862.706, 7862.869, 7873.943, 7874.006])
latency = np.array([7874.017, 210.871, 2043.537, 6.997, 979.021, 1947.493, 3.507, 891.648, 1777.862, 1.748, 0.161, 11.072, 0.060, 0.010])
event   = ["Interpreter", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer"]
colors = ["#1864ab", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9"]


ax1.barh(range(len(begin)),  latency, left=begin, color=colors)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (ms)")
plt.ylabel("Task")

data = [[ 0.004, 210.875, 2254.420, 2261.417, 3240.442, 5187.980, 5191.487, 6083.137, 7860.996, 7862.748, 7862.910, 7874.042, 7874.106],
        [ 210.869, 2043.543, 6.995, 979.024, 1947.536, 3.506, 891.648, 1777.855, 1.749, 0.160, 11.129, 0.061, 0.010],
        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]]

columns = ("Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer", "Layer")
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