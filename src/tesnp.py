import numpy as np
# a = np.array([[11,12], [13,14], [15,16]])
# b = np.array([[-11,-12,-13,-14,-15,-16], [-17,-18,-19,-20,-21,-22], [-23,-24,-25,-26,-27,-28]])
# f = np.array([[1,2,3],[4,5,6]])
# c = np.ndarray.flatten(a).reshape(6,1)
# d = np.ndarray.flatten(f).reshape(6,1)
# e = np.hstack((f,d))
# idx = c.argsort()[::-1]
# c = c[idx]
# b = b[:,idx]
# i = np.array([[2,0,0],[0,2,0],[0,0,2]])
# b[:,0] = i @ b[:,1]
# print(b,"\n",b[:,[1]])
#print(e)
i = np.array([[1,3],[-2,3],[1,9],[0,1],[1,7]])
c1 = i.T @ i
c2 = i @ i.T

e1, v1 = np.linalg.eig(c1)
e2, v2 = np.linalg.eig(c2)
print(e1,"\n",e2)


# print(np.array([np.mean(e, axis = 0)]))
# print(np.mean(e, axis = 1))
# print(e)