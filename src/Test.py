# encoding: utf-8
'''
Created on 2016年10月6日

@author: lenovo
'''
from idlelib.idle_test.test_idlehistory import line1, line2
from audioop import reverse
from doctest import Example
print'hello world'
print 'the quick brown fox','jumps over','the lazy dog'
print 300
print '100+300=',100+300#我想让他们之间没有空格怎么办
# name=raw_input('please enter your name:')
# print 'hello',name

#print absolute value of an integer
a=100
if a>=0:
    print a
else:
    print -a

#数据类型，计算机不仅可以处理数值，还可以处理文本，图形，音频、视频，网页等各种数据，不同数据要定义不同的数据类型
print 10e3
print -9.01
print 1.23e9
print 3/5
print "34"
print "I'm ok"
print 'I\'m \"ok\"'
print '\\\t'
print r'\\\\r\t'#表示内部不转义
print '''line1
line2
line3'''
print 'line1\nline2'
print True
print True and False
print True or False
print not True
print None
Answer=True
print Answer
x=10
x=x+2
print x
a='ABC'
b=a
a='XYZ'
print b
print 10.0/3

print ord('a')
print chr(65)
print u'中文'

classmates=['michael','Bob','Tracy']
print classmates
print len(classmates)
print classmates[0]
print classmates[len(classmates)-1]
print classmates[-1]
print classmates[-2]
print classmates[-3]
classmates.append('Adam')
print classmates
classmates.insert(1,'jack')
print classmates
classmates.pop()
print classmates
classmates.pop(1)
print classmates
classmates[1]='sarah'
print classmates
L=['Apple',123,True]
print L
s=['python','java',['asp','php'],'scheme']
print len(s)
p=['asp','php']
s=['python','java',p,'scheme']
L=[]
print len(L)
classmates=('michael','bob','tracy')
print classmates[0]
t=(1,2)
print t
t=(1)
print t
t=(1,)
print t
t=('a','b',['A','B'])
print t
t[2][0]='x'
t[2][1]='y'
print t
# t[1]=7 元组不能改变
# print t
for i in t:
    print i

sum=0
for x in [1,2,3,4,5,6]:
    sum=sum+x
    
print sum    
print range(5)
sum=0
for x in range(101):
    sum=sum+x
print sum

sum=0
n=99
while n>0:
    sum=sum+n
    n=n-2
print sum
#raw_input永远接收字符串,必须用int()
# birth=int(raw_input('birth:'))
# if birth<2000:
#     print '00前'
# else:
#     print '00后'
d={'michael':95,'bob':75,'tracy':85}
print d['michael']
d['adam']=67
print d
d['jack']=98
d['jack']=88
print d
print 'thomas' in d
#键值必须是保持不变，所以不能使用list
# key=[1,2,3]
# d[key]='a list'
# print d[key]

s=set([1,2,3])
print s
s.add(4)
print s
s.add(4)
print s
s.remove(4)
print s
s1=set([1,2,3])
s2=set([2,3,4])
print s1&s2
print s1 | s2

# s3=set([[1,2,3],1,3])
# print s3

a=['c','b','a']
a.sort()
print a 

a='abc'
print a.replace('a', 'A');
print a

print abs(-4)
print float(13.5)
print str(9.9)
print cmp(1,-1)
print unicode(100)
a=abs 
print a(-1)

def my_abs(x):
    if x>=0:
        return x
    else:
        return -x
    
print my_abs(-30)

def nop():
    pass #空函数

#pass可以用作占位符，比如说还没有想好如何写函数体

print my_abs('A')

def my_abs2(x):
    if not isinstance(x, (int,float)):
        raise TypeError('bad oprand type')#抛出错误
    if x>0:
        return x
    else:
        return -x
    
#print my_abs2("W")

#返回多个值

import math
def move(x,y,step,angle=0):
    nx=x+step*math.cos(angle)
    ny=y-step*math.sin(angle)
    return nx,ny

x,y=move(100,100,60,math.pi/6)
print x,y
#其实是一种假象，返回值是一个tuple。但是在语法上，返回一个tuple可以省略括号，而多个变量可以同时接受一个tuple，按位置赋予对应的值
r=move(100,100,60,math.pi/6)
print r

#函数定义之参数
def power(x,n=2):#有一个默认参数
    s=1;
    while n>0:
        n=n-1
        s=s*x
    return s

print power(3)
print power(5,3)
def enroll(name,gender,age=6,city='beijing'):
    print 'name:',name
    print 'gender:',gender
    print 'age:',age
    print 'city:',city
    
enroll('sarah','F')
enroll('mike', 'F', 7, 'henan')

def add_end(L=None):#设计成不变对象
    if L is None:
        L=[]
    L.append('end')
    return L

print add_end()

def clac(numbers):
    sum=0
    for n in numbers:
        sum=sum+n*n
    return sum

print clac([1,2,3])
print clac((1,2,3,4))

def clac2(*numbers):#定义成可变数组，就可以使用这种形式
    sum=0
    for n in numbers:
        sum=sum+n*n
    return sum

print clac2(1,2,3)
print clac2()#包括0个参数
#如果已经有了一个list或者tuple那该怎么做
nums=[1,2,3]
print clac2(nums[0],nums[1],nums[2])
print clac2(*nums)

def person(name,age,**kw):
    print 'name:',name,'age:',age,'other:',kw

person('bob', 14)
person('mike', 14,city='beijing',gender='F')

#如果已经有了一个字典
kw={'city':'beijing','gender':'F'}
person('jack',15,city=kw['city'],gender=kw['gender'])
person('jack',15,**kw)

#在Python中定义函数，可以用必选参数、默认参数、可变参数和关键字参数，这4种参数都可以一起使用，
#或者只用其中某些，但是请注意，参数定义的顺序必须是：
#必选参数、默认参数、可变参数和关键字参数。
def func(a, b, c=0, *args, **kw):
    print 'a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw

func(3, 4)
func(1,2,3)
func(1, 2, 3,'a','b')
func(1,2,3,'a','b',x=99)

args=(1,2,3,4,5)
kw={'x':99}
func(*args,**kw)

def fact(n):
    if n==1:
        return 1
    return n*fact(n-1)

print fact(6)
print fact(100)

#尾递归,单python并没有做尾递归的优化
def fact2(n):
    return fact_iter(n,1)

def fact_iter(num,product):
    if num==1:
        return product
    return fact_iter(num-1,num*product)

#print fact2(1000)

L=[]
n=1
while n<=99:
    L.append(n)
    n=n+2
    
print L

L=['michael','sarah','tracy','bob','jack']
print [L[0],L[1],L[2]]

r=[]
n=3
for i in range(n):
    r.append(L[i])
print r

print L[0:3]
print L[1:3]

#倒切片
print L[-2:-1]

L=range(100)
print L

print L[:10]
print L[-10:]
print L[10:20]
print L[:10:2]
print L[::5]

print L[:]

print (0,1,2,3,4,5)[:3]
L=(0,1,2,3,4,5,6)
print L[-2:]

print 'ABCDEFG'[:3]
print 'ABCDEFG'[::2]

d={'a':1,'b':2,'c':3}
for value in d.itervalues():
    print value
    
from collections import Iterable
print isinstance('abc', Iterable)

#如果要对list实现类似Java那样的下标循环怎么办，python提供了enumerate函数可以把list变成索引——元素对
#enumrate
for i,value in enumerate(['A','B','C']):
    print i,value
    
for x,y in [(1,1),(2,4),(3,9)]:
    print x,y

#列表生成式,List Comprehensions
print range(1,11)
#如果要成[1*1,2*2,3*3...,10*10]
L=[]
for x in range(1,11):
    L.append(x*x)
print L
#把要生成的x*x放在前面，后面跟for循环
print [x*x for x in range(1,12)]

#还可以跟上条件语句
print [x*x for x in range(1,12) if x%2==0]
#还可以使用两层循环，可以生成全排列
print [m+n for m in 'ABC' for n in 'XYZ']
print [m+n+l for m in 'ABC' for n in 'XYZ' for l in 'LMN']

import os#导入os模块
print [d for d in os.listdir('.')]

d={'x':'A','y':'B','z':'C'}
for k,v in d.iteritems():
    print k,'=',v
    
print [k+'='+v for k,v in d.iteritems()]

L=['Hello','World','IBM',"Apple"]
print [s.lower() for s in L]
x='abc'
y=123
print isinstance(x, str)
print isinstance(y, str)
#如果数据参差不齐
L=['Hello','World',18,'Apple',None]
print [s.lower() for s in L if isinstance(s, str)]

#通过列表生成式，我们可以直接创建一个列表，但是受到内存限制，列表容量肯定是有限的
#若果只访问前面一些元素，那这个大list占用的内存将会大大浪费
#如果列表元素尅按照某种算法推算出来，就不必创建完整的list
#这种一边循环一边计算的机制，称为生成器（Generator)

L=[x*x for x in range(10)]
print L
#生成器和生成式的区别仅是一个括号
g=(x*x for x in range(10))
print g
print g.next()
print g.next()
print g.next()
print g.next()
for n in g:
    print n
    
def fib(max):
    n,a,b=0,0,1
    while n<max:
        print b
        a,b=b,a+b
        n=n+1
        
fib(6)

#定义generator的另一种方法
def fib2(max):
    n,a,b=0,0,1
    while n<max:
        yield b#yield 关键字，生成generator方式
        a,b=b,a+b
        n=n+1
        
print fib2(6)
for n in fib2(6):
    print n
    
def odd():
    print 'step1'
    yield 1
    print 'step2'
    yield 3
    print 'step3'
    yield 5
o=odd()
print o.next()
print o.next()
print o.next()

#在计算机的层次上，cpu执行董事加减乘除的指令代码，以及各种条件判断和跳转语句，所以
#汇编语言是最贴近计算机的语言

#高阶函数
print abs
#变量可以指向函数
a=abs
print a
print abs(20)
#函数式编程，把函数作为参数传入，这样的函数称为 高阶函数
def add(x,y,f):
    return f(x)+f(y)#美

print add(-5,6,abs)

#map/reduce
#python内置了map()和reduce()函数
def f(x):
    return x*x

print map(f,[1,2,3,4,5,6,7,8,9])#map传入的第一个参数是f，即函数对象本身

print map(str,[1,2,3,4,5,6,7,8,9])

def add2(x,y):
    return x+y

print reduce(add2, [1,3,5,7,9])

def fn(x,y):
    return x*10+y

print reduce(fn, [1,3,5,7,9])

def char2num(s):
    return {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]

print char2num('1')
print reduce(fn, map(char2num, '13579'))

#函数式编程加map/reduce
def str2int(s):
    def fn(x,y):
        return x*10+y
    def char2num(s):
        return {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]
    return reduce(fn, map(char2num, s))

print str2int('123')

def str2int_new(s):
    return reduce(lambda x,y:x*10+y, map(lambda s:{'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s], s))

inTter=['adam','LISA','barT']
regNames=lambda iter:map((lambda inStr:inStr.capitalize()),iter)
print regNames(inTter)

#filter

def is_odd(n):
    return n%2==1

print filter(is_odd, [1,2,3,4,5,6,7])

def not_empty(s):
    return s and s.strip()

print filter(not_empty, ['A','','B',None,'C',' '])
def is_prime(n):
    if n==1:
        return True
    for i in range(2,n):
        if n%i==0:
            return True
    return False

print filter(is_prime, range(1,101))

#sorted
print sorted([36,5,13,9,21])
#他还可以接收一个比较函数来实现自定义的排序，比如还可以倒序排序
def reversed_cmp(x,y):
    if(x>y):
        return -1
    if(x<y):
        return 1
    return 0
#这样就可以实现倒序排序
print sorted([36,5,12,9,21],reversed_cmp)#记住只是函数名，没有括号
#字符串默认情况下是按照ascii码的大小进行比较的，小写的要比大写的大
print sorted(['bob','about','Zoo','Credit'])
#现在我们提出应该忽略大小写
def cmp_ignore_case(s1,s2):
    u1=s1.upper()
    u2=s2.upper()
    if u1<u2:
        return -1
    if u1>u2:
        return 1
    return 0

print sorted(['bob','about','Zoo','Credit'],cmp_ignore_case)
#高阶函数的抽象能力非常强大，可以使代码保持的非常整洁

#返回函数，高阶函数除了可以接受函数作为参数外，还可以把函数作为结果值返回

def clac_sum(*args):
    ax=0
    for n in args:
        ax=ax+n
    return ax
print clac_sum(1,3,5,7,9)
#这种称为‘闭包（closure）’的程序结构拥有极大的威力
def lazy_sum(*args):
    def sum():
        ax=0
        for n in args:
            ax=ax+n
        return ax
    return sum
f=lazy_sum(1,3,5,7,9)
print f()
f1=lazy_sum(1,2,3)
f2=lazy_sum(1,2,3)
print f1==f2
#闭包使用
def count():
    fs=[]
    for i in range(1,4):
        def f():
            return i*i
        fs.append(f)
    return fs

f1,f2,f3=count()
print f1(),f2(),f3()

def count2():
    fs=[]
    for i in range(1,4):
        def f(j):
            def g():
                return j*j
            return g
        fs.append(f(i))
    return fs

f1,f2,f3=count2()
print f1(),f2(),f3()

#匿名函数
print map(lambda x:x*x, [1,2,3,4])
# 匿名函数也是一个函数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数
f=lambda x:x*x
print f
print f(5)

def build(x,y):
    return lambda:x*x+y*y

print build(2, 3) 
f=build(3, 3)
print f()

def now():
    print '2013-12-25'
    
f=now
f()
print now.__name__
print f.__name__

#装饰器，假设我们现在要增强now()的功能，比如在函数调用前后自动打印日志，
#但又不希望修改弄完()函数的定义，这种在代码运行期间动态增加动能的方式，
#称之为“装饰器”（decorator）

# def log(func):
#     def wrapper(*args,**kw):
#         print 'call %s():' % func.__name__
#         return func(*args,**kw)
#     return wrapper()
# @log
# def now2():
#     print '2013-12-25'
#     
# now2()

#偏函数
# print int('12345',base=8)
# 
# def int2(x,base=2):
#     return int(x,base)
# 
# print int2('1000000')
# 
# import functools
# int2=functools.partial(int, base=2)

#模块
import sys

def test():
    args=sys.argv
    if len(args)==1:
        print 'hello world'
    elif len(args)==2:
        print 'hello,%s!' % args[1]
    else:
        print 'too many arguments'
        

test()
from numpy import *
print zeros((2,3))

import operator
d={'a':1,'b':3,'c':2}
s= sorted(d.iteritems(),key=operator.itemgetter(1),reverse=True)
print s[0][0]

d=[[1,2,3],[2,3,4],[3,4,5]]
t=mat(d)
print t[1,:]
returnVect = zeros((1,1024))
print returnVect
returnVect[0,6]=int(6)
print returnVect
from math import log
import operator
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

myDat,labels=createDataSet()

print calcShannonEnt(myDat)
myDat[0][-1]='maybe'
print myDat
print calcShannonEnt(myDat)

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

myDat,labels=createDataSet()
print splitDataSet(myDat, 0, 1)

t=myDat[0]
print t

featList=[example[2] for example in myDat]
print featList

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet, i, value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature


print myDat
print chooseBestFeatureToSplit(myDat)

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)#如果已经处理了所有属性，类别标签依然不是唯一的，此时我们需要决定如何定义该叶子节点，通常采用多数表决法
    bestFeat = chooseBestFeatureToSplit(dataSet)#根据信息增益得到的最好的划分属性
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])#del用于list操作，用于删除一个或连续几个元素，属性用完就不需要了
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree   

myDat,labels=createDataSet()
myTree=createTree(myDat, labels)
print myTree
    
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            



