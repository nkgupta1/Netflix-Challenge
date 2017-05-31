#!/usr/bin/env python3

# CF files produce stupid output. This file fixes that.
probe_length = 1374739
qual_length = 2749898

g = open("cf_probe_preds2.dta", "r")
f = open("../../../../data/um/probe_all.dta", "r")
h = open("cf_clean_probe.dta", "w")

most_rec_line = g.readline()
for line in f:
    line_g = most_rec_line

    lst_pred = line.split()
    lst_probe = line_g.split()

    u_pred = int(lst_pred[0])
    m_pred = int(lst_pred[1])
    r_pred = float(lst_pred[3])

    if (len(lst_probe) == 3):
        u_probe = int(lst_probe[0])
        m_probe = int(lst_probe[1])
        r_probe = float(lst_probe[2])
    else:
        u_probe = -1
        m_probe = -1

    if ((u_pred == u_probe) and (m_pred == m_probe)):
        h.write(str(r_probe)+'\n')
        most_rec_line = g.readline()
    else:
        h.write(str(3.6)+'\n')

g = open("cf_qual_preds.dta", "r")
f = open("../../../../data/um/qual_all.dta", "r")
h = open("cf_clean_qual.dta", "w")

most_rec_line = g.readline()
for line in f:
    line_g = most_rec_line

    lst_pred = line.split()
    lst_probe = line_g.split()

    u_pred = int(lst_pred[0])
    m_pred = int(lst_pred[1])
    r_pred = float(lst_pred[3])

    if (len(lst_probe) == 3):
        u_probe = int(lst_probe[0])
        m_probe = int(lst_probe[1])
        r_probe = float(lst_probe[2])
    else:
        u_probe = -1
        m_probe = -1

    if ((u_pred == u_probe) and (m_pred == m_probe)):
        h.write(str(r_probe)+'\n')
        most_rec_line = g.readline()
    else:
        h.write(str(3.6)+'\n')


print("All is good!")
