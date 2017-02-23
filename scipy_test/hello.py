import numpy as np


#exercise

#1. rank(5) col(3)
'''
mlist = [[i+1] for i in range(5)]
for index in range(10):
    mlist[index%5].append(index+6)
print mlist

marr = np.array(mlist)  # conversion from list to array
print marr
'''
marr = np.arange(1,16).reshape(3,5).transpose()
print marr
sarr = marr[1::2,:] # slicing & substracting for array
print sarr

#2. elementalwise divide a with b
a = np.arange(25).reshape(5,5)
b = np.array([1., 5, 10, 15, 20])
print a/b

#3.
oarr = np.random.rand(10,3)
print oarr

#marr = np.abs(oarr - np.array([0.5, 0.5, 0.5]))
marr = np.abs(oarr - np.full(3, 0.5))
print np.full(3, 0.5) # 1d array & fixed value (2d array: np.full((row,col), value)

ilist = np.argmin(marr, axis=1)
sarr = np.array([oarr[i, ilist[i]] for i in range(10)])
print sarr

#4 image handling
from scipy import misc
import matplotlib.pylab as plt

face = misc.face(gray=True)
plt.imshow(face)

ny, nx = face.shape
y, x = np.ogrid[0:ny, 0:nx] # generate multi-demensional meshgrid
xcenter, ycenter = (660, 300)
xvalue = (x - xcenter)**2
yvalue = (y - ycenter)**2
print xvalue
print xvalue.shape

mask = (xvalue + yvalue) > 230**2
print mask

xvalue = np.array([[1,2]])
yvalue = xvalue.transpose()

print xvalue
print yvalue

# range vs. xrange
# x in (x)ranage: dtype = list(xrange)

def f(a,b,c):
    return a**b - c

print "integral"
print f(2,2,2)

na = np.ogrid[0:1:24j]
nb = np.ogrid[0:1:12j]
nc = np.ogrid[0:1:6j]
# 24 12 6
na3 = na.reshape((24,1,1))
nb3 = nb.reshape((1,12,1))
nc3 = nc.reshape((1,1,6))
print f(na3, nb3, nc3).mean()   # 0.1931 ...

print str(na3.ravel().data)


'''
from matplotlib import pyplot as plt
data = np.loadtxt('/Users/ahnyoungho/GitHub/scipy-lecture-notes/data/populations.txt')
print data.T
year, hares, lynxes, carrots = data.T

plt.axes([0.2, 0.1, 0.5, 0.8])
plt.plot(year, hares, year, lynxes, year, carrots)
plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05, 0.5))
#plt.show()

print data
sub_pop = data[:,1:] # without year
print sub_pop.mean(axis=0)
print np.argmax(sub_pop, axis=1)
print np.argmin(sub_pop, axis=1)
'''

'''
str = "hello world!"
num1 = 90
num2 = 100
sum = num1 + num2
print str + str
print 'hello' ' ' 'world!'
print str[0]
print len(str) # 0:len-1
print str[len(str)-1]

mlist = ['cat','1','dog','7','end']
print mlist
for slist in mlist[:4]:
  print (slist,len(slist))

print ('length', len(mlist))

for i in range(len(mlist)):
  print mlist[i]

for n in range(2,10):
  for x in range(2,n):
    if n % x == 0:
      print (n, 'equal', x, '*', n/x)
      break
  else:
    print (n, 'is a prime number')

def mfunc(n):
  a, b = 0, 1
  while a < n:
    print a
    a, b = b, a+b

mfunc(1000)

#sqList = []
#sqList.append(2**2)
#for n in range(3,10):
#    sqList.append(n**2)
sqList = [x**2 for x in range(10)]
print sqList

sqList = [(x,y) for x in [1,2,3] for y in [1,5,6]]
print sqList
sqList.reverse()
print sqList

str = "hello world"
strList = str.split()
print strList

strList = list(str)
print strList

# data types - set(): {}, list():[], '' or "", (,,,)
print "numpy test"
np_array2d = np.zeros((3,4)) # (row,col)
print np_array2d
np_arr2d_rand = np.random.rand(3,4)
print np_arr2d_rand
print np_arr2d_rand[1,2] # (row, col) where 0<= row, col < Max

wines = np.array([[1,2],[4,5],[3,3]])
print (wines+np.array([1,1])) # broadcasting
np_arr1d_rand = np.random.rand(1,1)
rwines = wines + np_arr1d_rand
print rwines # broadcasting with 2d rand()
print "test axis"
print wines
print wines.sum(axis=0) # sum, dimension(rank): {axis| 0: vertical, 1: horizental, ... < rank}
print wines[:2,:].sum(axis=1) # sum & partitioning

boolean_arr = rwines[:,0] > 2 # result: boolean array
print boolean_arr
print wines[boolean_arr,1]
print wines.shape
print np.transpose(wines).shape
print wines.ravel() # flattern 2d to 1d

rwines += wines
print rwines
'''
