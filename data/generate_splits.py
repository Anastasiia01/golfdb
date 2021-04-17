from scipy.io import loadmat
import pandas as pd
import os

"""cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))"""

x = loadmat('handwash.mat')
l = list(x['handwash'][0])
d = dict()
for idx, k in enumerate(l):
    d["{:3d}".format(idx)] = list(l[idx])
df = pd.DataFrame(d).T
df.columns = ["video_name", "events", "split", "view", "total_frames"]

# data format cleansing by removing extra bracets around data
df['video_name'] = df['video_name'].apply(lambda x: x[0][0])
df['events'] = df['events'].apply(lambda x: x[0])
df['split'] = df['split'].apply(lambda x: x[0][0])
df['view'] = df['view'].apply(lambda x: x[0])
df['total_frames'] = df['total_frames'].apply(lambda x: x[0][0])

df.index = df.index.astype(int)
df.to_pickle('handwash.pkl')

for i in range(1, 5):
    val_split = df.loc[df['split'] == i]
    # Next two code lines change val_split.index from Int64Index([0 ... n], dtype='int64')
    # to RangeIndex(start=0, stop=n+1, step=1) where n = number of rows
    val_split = val_split.reset_index()
    val_split = val_split.drop(columns=['index']) 
    val_split.to_pickle("val_split_{:1d}.pkl".format(i))

    train_split = df.loc[df['split'] != i]
    train_split = train_split.reset_index()
    train_split = train_split.drop(columns=['index'])
    train_split.to_pickle("train_split_{:1d}.pkl".format(i))

#print("Number of unique videos: {:3d}".format(len(df['video_name'].unique())))
print("Number of annotations: {:3d}".format(len(df.video_name)))