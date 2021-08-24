import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.suptitle('Performance')

begin   = np.array([0.000, 0.004, 4215.811, 5739.313, 8052.235, 15255.614, 16398.968, 18941.941, 22898.829, 24598.854, 28396.555, 28407.363, 32357.799, 33434.675, 35018.045, 37725.256, 39457.527, 42098.158, 42102.656, 44809.967, 44983.608, 46311.063, 48995.891, 49329.523, 51981.241, 51983.491, 54669.451, 55003.112, 57654.838, 57657.090, 60343.188, 61149.146, 64862.236, 70106.154, 71233.454, 76427.597, 76430.748, 81674.764, 82801.674, 87995.753, 87998.904, 93242.804, 93525.558, 95751.745, 99589.415, 99999.948, 103815.766, 103817.122, 107654.444, 108067.120, 111882.951, 111884.306, 115722.153, 116132.477, 119947.754, 119949.110, 123787.218, 123973.026, 130332.745, 137425.238, 137439.931, 137440.090, 137514.924])
latency = np.array([137515.235, 4215.801, 1523.497, 2312.918, 7203.376, 1143.349, 2542.969, 3956.884, 1700.020, 3797.697, 10.805, 3950.432, 1076.872, 1583.367, 2707.207, 1732.268, 2640.627, 4.496, 2707.307, 173.637, 1327.452, 2684.824, 333.628, 2651.715, 2.247, 2685.957, 333.658, 2651.722, 2.249, 2686.095, 805.955, 3713.086, 5243.914, 1127.296, 5194.139, 3.148, 5244.013, 1126.905, 5194.076, 3.148, 5243.896, 282.750, 2226.184, 3837.666, 410.529, 3815.815, 1.354, 3837.318, 412.672, 3815.827, 1.353, 3837.844, 410.319, 3815.274, 1.353, 3838.104, 185.804, 6359.716, 7092.489, 14.690, 0.157, 74.830, 0.309])
event   = ["Interpreter", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX"]
colors = ["#1864ab", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9", "#4a98c9"]


ax1.barh(range(len(begin)),  latency, left=begin, color=colors)
ax1.grid(linestyle = ':')


plt.sca(ax1)
plt.yticks(range(len(begin)), event)
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.tick_params(axis='both', which='minor', labelsize=1)

plt.xlabel("Schedule (ms)")
plt.ylabel("Task")

data = [[ 0.004, 4215.811, 5739.313, 8052.235, 15255.614, 16398.968, 18941.941, 22898.829, 24598.854, 28396.555, 28407.363, 32357.799, 33434.675, 35018.045, 37725.256, 39457.527, 42098.158, 42102.656, 44809.967, 44983.608, 46311.063, 48995.891, 49329.523, 51981.241, 51983.491, 54669.451, 55003.112, 57654.838, 57657.090, 60343.188, 61149.146, 64862.236, 70106.154, 71233.454, 76427.597, 76430.748, 81674.764, 82801.674, 87995.753, 87998.904, 93242.804, 93525.558, 95751.745, 99589.415, 99999.948, 103815.766, 103817.122, 107654.444, 108067.120, 111882.951, 111884.306, 115722.153, 116132.477, 119947.754, 119949.110, 123787.218, 123973.026, 130332.745, 137425.238, 137439.931, 137440.090, 137514.924],
        [ 4215.801, 1523.497, 2312.918, 7203.376, 1143.349, 2542.969, 3956.884, 1700.020, 3797.697, 10.805, 3950.432, 1076.872, 1583.367, 2707.207, 1732.268, 2640.627, 4.496, 2707.307, 173.637, 1327.452, 2684.824, 333.628, 2651.715, 2.247, 2685.957, 333.658, 2651.722, 2.249, 2686.095, 805.955, 3713.086, 5243.914, 1127.296, 5194.139, 3.148, 5244.013, 1126.905, 5194.076, 3.148, 5243.896, 282.750, 2226.184, 3837.666, 410.529, 3815.815, 1.354, 3837.318, 412.672, 3815.827, 1.353, 3837.844, 410.319, 3815.274, 1.353, 3838.104, 185.804, 6359.716, 7092.489, 14.690, 0.157, 74.830, 0.309],
        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]]

columns = ("CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D", "CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX")
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