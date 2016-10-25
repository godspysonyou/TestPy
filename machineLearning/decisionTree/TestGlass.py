# encoding: utf-8
'''
Created on 2016年10月18日

@author: lenovo
'''
import trees
import treePlotter
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=trees.createTree(lenses,lensesLabels)
print lensesTree
treePlotter.createPlot(lensesTree)