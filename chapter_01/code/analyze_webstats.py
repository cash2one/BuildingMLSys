import os
import scipy as sp
import matplotlib.pyplot as plt

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data")
data = sp.genfromtxt(os.path.join(data_dir, "web_traffic.tsv"), delimiter="\t")
print(data[:10])

# all examples will have three classes in this file
colors = ['g', 'k', 'b', 'm', 'r']
line_style_list = ['-', '-.', '--', ':', '-']

x = data[:, 0]
y = data[:, 1]
# print("Number of invalid entries:", sp.sum(sp.isnan(y)))
print("无效实例的数量:", sp.sum(sp.isnan(y)))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


def plot_models(x, y, models, filename, mx=None, ymax=None, xmin=None):
    # plot input data
    """
    绘制输入的数据

    :param x:
    :param y:
    :param models:
    :param filename: 图片的文件名
    :param mx:
    :param ymax:
    :param xmin:
    :return:
    """
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks(
        [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, line_style_list, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(filename)


# first look at the data
plot_models(x, y, None, os.path.join("..", "1400_01_01.png"))

# create and plot models
fp1, res, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

# print("Model parameters: %s" % fp1)
print("模型参数: %s" % fp1)
# print("Error of the model:", res)
print("模型误差:", res)
f1 = sp.poly1d(fp1)
f2 = sp.poly1d(sp.polyfit(x, y, 2))
f3 = sp.poly1d(sp.polyfit(x, y, 3))
f10 = sp.poly1d(sp.polyfit(x, y, 10))
f100 = sp.poly1d(sp.polyfit(x, y, 100))

plot_models(x, y, [f1], os.path.join("..", "1400_01_02.png"))
plot_models(x, y, [f1, f2], os.path.join("..", "1400_01_03.png"))
plot_models(
    x, y, [f1, f2, f3, f10, f100], os.path.join("..", "1400_01_04.png"))
# exit()
# fit and plot a model using the knowledge about inflection point
inflection = int(3.5 * 7 * 24)
print(inflection)

xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

plot_models(x, y, [fa, fb], os.path.join("..", "1400_01_05.png"))


def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)

# print("Errors for the complete data set:")
print("整个数据集的误差:")
for f in [f1, f2, f3, f10, f100]:
    # print("Error d=%i: %f" % (f.order, error(f, x, y)))
    print("误差 d=%i: %f" % (f.order, error(f, x, y)))

# print("Errors for only the time after inflection point")
print("在拐点后的误差")
for f in [f1, f2, f3, f10, f100]:
    # print("Error d=%i: %f" % (f.order, error(f, xb, yb)))
    print("误差 d=%i: %f" % (f.order, error(f, xb, yb)))

# print("Error inflection=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))
print("误差拐点=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))


# extrapolating into the future
plot_models(
    x, y, [f1, f2, f3, f10, f100], os.path.join("..", "1400_01_06.png"),
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# print("Trained only on data after inflection point")
print("只训练在拐点后的数据")
fb1 = fb
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
fb3 = sp.poly1d(sp.polyfit(xb, yb, 3))
fb10 = sp.poly1d(sp.polyfit(xb, yb, 10))
fb100 = sp.poly1d(sp.polyfit(xb, yb, 100))

# print("Errors for only the time after inflection point")
print("在拐点后的误差")
for f in [fb1, fb2, fb3, fb10, fb100]:
    # print("Error d=%i: %f" % (f.order, error(f, xb, yb)))
    print("误差 d=%i: %f" % (f.order, error(f, xb, yb)))

plot_models(
    x, y, [fb1, fb2, fb3, fb10, fb100], os.path.join("..", "1400_01_07.png"),
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# separating training from testing data
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))

# print("Test errors for only the time after inflection point")
print("在拐点后的测试误差")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    # print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))
    print("误差 d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

plot_models(
    x, y, [fbt1, fbt2, fbt3, fbt10, fbt100], os.path.join("..",
                                                          "1400_01_08.png"),
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

from scipy.optimize import fsolve
print(fbt2)
print(fbt2 - 100000)
reached_max = fsolve(fbt2 - 100000, 800) / (7 * 24)
# print("100,000 hits/hour expected at week %f" % reached_max[0])
print("预计在 %f 周后达到100,000次/小时的点击量" % int(round(reached_max[0])))
print("预计在 %f 周后达到100,000次/小时的点击量" % reached_max[0])
