import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb

# data = {'method': ['G.A.', 'G.A.', 'G.A.', 'G.A.', 'Grad.', 'Grad.', 'Grad.', 'Grad.', 'COMs', 'COMs', 'COMs', 'COMs', 'Oracle', 'Oracle', 'Oracle', 'Oracle', 'Dataset', 'Dataset', 'Dataset', 'Dataset'], 'percentile': ['100th', '90th', '80th', '50th', '100th', '90th', '80th', '50th', '100th', '90th', '80th', '50th', '100th', '90th', '80th', '50th', '100th', '90th', '80th', '50th'], 'score': [98.715668, 95.299324, 94.109795, 81.258179, 99.192497, 96.780334, 95.387184, 90.416832, 99.084106, 95.769913, 92.712051, 83.728065, 98.228226, 89.240257, 82.387688, 71.650223, 72.350487, 71.344887, 70.110947, 64.836395]}

data = {'method': ['G.A.', 'Grad.', 'COMs', 'Oracle', 'Dataset'],
        '100th': [98.715668, 99.192497, 99.084106, 98.228226, 72.350487],
        '90th': [95.299324, 96.780334, 95.769913, 89.240257, 71.344887],
        '80th': [94.109795, 95.387184, 92.712051, 82.387688, 70.110947],
        '50th': [81.258179, 90.416832, 83.728065, 71.650223, 64.836395]}
df = pd.DataFrame(data)

palette = sns.color_palette("PuBu_r")
# sns.catplot(data=df, x='method', y='score', hue='percentile', kind='bar', palette=palette)
sns.barplot(data=df, x='method', y='100th', color=palette[0], width=0.5)
sns.barplot(data=df, x='method', y='90th', color=palette[1], width=0.5)
sns.barplot(data=df, x='method', y='80th', color=palette[2], width=0.5)
sns.barplot(data=df, x='method', y='50th', color=palette[3], width=0.5)
plt.xticks(rotation=0)
plt.ylabel('score')
plt.savefig('./hist.jpg', dpi=400)
plt.close()


