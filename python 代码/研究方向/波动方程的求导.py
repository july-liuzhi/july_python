
from sympy import*

def kkk(u_str: str, x_str: str, t_str: str):
    x = symbols(x_str)
    t = symbols(t_str)
    u = sympify(u_str)
    u1 = lambdify((x, t), u, 'numpy')
    u2 = lambdify((x, t), diff(u, t, 2) - diff(u, x, 2), 'numpy')
    # u2 = simplify(u2)
    return u2

k = kkk("(sin(pi*(x-t))+sin(pi*(x+t)))/2 - (sin(pi*(x-t))+sin(pi*(x+t)))/(2*pi)", "x", "t")
print(k(2,5))
