#!/usr/bin/env python3
import os

date = 5
submitter = 6
description = 8

files = ["dc1l", "lr1dr04vc05v17a-t360", "cdma", "fastxgemm-n3r23s5t6", "pb-gfrd-pnc", "genus-sym-grafo5708-48", "neos-4295773-pissa", "ger50-17-trans-dfn-3t", "set3-09", "neos-3322547-alsek", "fhnw-schedule-paira400", "neos-3615091-sutlej", "polygonpack5-15", "core2586-950", "polygonpack3-15", "adult-max5features", "ex1010-pi", "tokyometro", "rmine21","n370b", "n3705", "rvb-sub", "dano3mip", "neos-4292145-piako", "tpl-tub-ws1617", "ns1849932", "cmflsp40-36-2-10", "ns1856153", "n3700", "ger50-17-ptp-pop-6t", "neos-5041822-cockle", "core4872-1529", "fastxgemm-n3r21s3t6", "neos-5266653-tugela", "shiftreg5-1", "fastxgemm-n3r22s4t6", "ns1904248", "rwth-timetable", "neos-4290317-perth", "tpl-tub-ss16", "lr1dr02vc05v8a-t360", "siena1", "neos-3372571-onahau", "scpj4scip", "comp16-3idx", "zeil", "t1717", "fhnw-schedule-pairb200", "fhnw-schedule-pairb400", "r4l4-02-tree-bounds-50", "d20200", "rocII-8-11", "n3709", "neos-4355351-swalm", "snp-04-052-052", "cmflsp60-36-2-6", "rfds-4-days", "splan1", "cdc7-4-3-2", "sct1", "fhnw-schedule-paira200", "ds-big", "xmas10-2", "fhnw-schedule-paira100", "comp12-2idx", "ns1690781", "dlr1", "hgms62", "sorrell7", "nsr8k", "snip10x10-35r1budget17", "neos-3594536-henty", "ger50-17-ptp-pop-3t", "ns2122698", "momentum3", "t1722", "supportcase38", "core4284-1064", "ns1631475", "scpk4", "rmine25", "sorrell4", "shs1042", "n3707", "supportcase2", "z26", "gmut-76-50", "allcolor10"]

files2 = files.copy()

SOLUTIONDIR = "/scratch/opt/bzfkress/miplib2017/Solutions/solutions"

count = 0
for instance in files:
    found = False
    for n in os.listdir( os.path.join( SOLUTIONDIR, instance ) ):
        with open( os.path.join(SOLUTIONDIR, instance, n, "info.yaml"), "r+" ) as f:
            lines = []
            for line in f:
                lines.append(line)

            if lines[date] != 'date : 2020-02-04\n':
                continue
            if lines[submitter] != 'submitter : "Edward Rothberg"\n':
                continue
            if lines[description] != '  Found with Gurobi 9.0\n':
                continue
            found = True

            # set correct values
            lines[date] = 'date : 2019-12-13\n'
            lines[description] = '  Obtained with Gurobi 9.0\n'
            f.seek(0)
            f.writelines(lines)
    if found:
        files2.remove(instance)
        count += 1


print(count)
print(len(files))
print(files2)