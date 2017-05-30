#!/usr/bin/env python3

# CF files produce stupid output. This file fixes that.
probe_length = 1374739
qual_length = 2749898

f = open("cf_probe_preds.dta", "r")

# BELOW IS OUTDATED
for i in range(probe_length):
    line = f.readline()
    if (len(line) <= 2):
        x = float(line)
    else:
        x = float(line.split()[1][:-2])

print("All is good!")
