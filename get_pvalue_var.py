#coding:utf-8
import os
path="E:/code/addr_datat/抗菌药占比/tmptmp.txt"
pathout="E:/code/addr_datat/抗菌药占比/tmpPv.txt"

lines=[]
with open(path) as fin,open(pathout,'w') as fout:
    for line in fin:
        pv=line.strip().split()[-3]
        pv=float(pv)
        if pv <0.05:
            lines.append(line)

def sort_key(line):
    num=line.strip().split()[1]
    num=float(num)
    return num
def fd():
    global lines
    lines=sorted(lines,key=sort_key,reverse=True)
    patho="E:/code/addr_datat/抗菌药占比/tmpPvtmp.txt"
    with open(patho,'w') as fout:
        for line in lines:
            fout.write(line)