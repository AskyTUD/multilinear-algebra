from casadi import *
# x = MX.sym('x',2)
# y = MX.sym('y')
# f = Function('f',[x,y],\
#       [x,sin(y)*x],\
#       ['x','y'],['r','q'])

x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z')
p = SX.sym('p')
pp = SX.sym('qq')
nlp = {'x':vertcat(x,y,z), 'f':x**2+p*z**2, 'g':z+(pp-x)**2-y, 'p': vertcat(p, pp)}
# {'ipopt.linear_solver': 'ma27'}
S = nlpsol('S', 'ipopt', nlp, {'ipopt.linear_solver': 'ma97'})

print(S)

r = S(x0=[2.7, 3.0, 0.75], lbg=0, ubg=0, p=[100, 1])
x_opt = r['x']
print('x_opt: ', x_opt)
