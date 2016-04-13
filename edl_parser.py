import sys
filename = sys.argv[1]

drop_frame = True
skip = 4
start = 3

with open(filename, mode='r') as f:
    lines = f.readlines()

if 'NON' in lines[1]:
    drop_frame = False
    start = 4

times = [line.split()[-1] for line in lines[start::skip]]
feeds = [line.split()[-1] for line in lines[start+1::skip]]

for time, feed in zip(times, feeds):
    print(time, feed)
