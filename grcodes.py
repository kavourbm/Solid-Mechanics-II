import sys
import numpy as np
import sympy as sp
import scipy
import numpy.linalg as lg   
import math as ma

import inspect
import matplotlib.pyplot as plt      # for plotting figures
plt.rcParams['figure.dpi'] = 500
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.figsize':(6,4)})
#plt.rcParams['axes.linewidth'] = 0.3

def printx(name):                            
    """ This prints out both the name and its value together.  
    usage: name = 88.0;  printx('name') 
    Ref: https://stackoverflow.com/questions/32000934/
    python-print-a-variables-name-and-value"""

    frame = sys._getframe(1)
    print(name, '=', repr(eval(name, frame.f_globals, frame.f_locals)))

#x = 1 
#printnve('x')

#def printnv(name): 
#    print (name, '=', repr(eval(name)))

def principalS(S):
    '''Compute the principal stresses and their directions.
    inputs: 
       S: given stress tensor, numpy array
    return: 
       principal stresses (eigenValues), their direction cosines
       (eigenVectors) ranked by its values. Right-hand-rule is enforced
    '''
    eigenValues, eigenVectors = lg.eig(S)  

    #Sort in order
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    print('Pricipal stress (Eigenvalues):\n',eigenValues,'\n')
    
    # make the first element in the first vector positive (optional):
    #eigenVectors[0,:] = eigenVectors[0,:]/np.sign(eigenVectors[0,0])

    # Determine the sign for given eigenVector-1 and eigenVector-3
    eigenVectors[:,2] = np.cross(eigenVectors[:,0], eigenVectors[:,1])

    angle = np.arccos(eigenVectors[0,0])*180/np.pi       # in degree 
    print(f'Principal stress directions:\n{eigenVectors}\n')
    print(f"Possible angles (n1,x)={angle}∘ or {180-angle}∘")
    
    return eigenValues, eigenVectors


def M_eigen(stressMatrix):
    '''Compute the eigenvalues and eigenvectors using the stress matrix 
    Input stressMatrix: the stress matrix with 9 components.
    Return: the eigenvaluesare shorted in descending
    order, and the eigenvectors will be sorted accordingly. The sign 
    for eigenVectors are also adjusted based on the right-hand rule. 
    '''

    eigenValues, eigenVectors = lg.eig(stressMatrix)  

    #Sort in order
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # Determine the sign for given eigenVector-1 and eigenVector-3
    eigenVectors[2,:] = np.cross(eigenVectors[0,:], eigenVectors[1,:])
    return eigenValues, eigenVectors

def von_Mises(e):                    
    '''Compute the von Mises stress using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
    '''
    vonMises=np.sqrt(((e[0]-e[1])**2+(e[0]-e[2])**2+(e[1]-e[2])**2)/2.)
    return vonMises

def Oct_stress(e):                    
    '''Compute the octahedral shear stress using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
    '''
    sigma_oct= np.sum(e)/3. 
    tau_oct=np.sqrt((e[0]-e[1])**2+(e[0]-e[2])**2+(e[1]-e[2])**2)/3. 
    return sigma_oct, tau_oct

def energy_stress(e,nu):                    
    '''Compute the strain-energy stress using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
         nu: Poission's ratio.
    '''
    U_stress=np.sqrt(e[0]**2+e[1]**2+e[2]**2-2*nu*(e[1]*e[2]+
                     e[0]*e[2]+e[0]*e[1]))
    return U_stress

def M_invariant(e):                    
    '''Compute the 3 stress/strain invariants using principal stresses
    Input e: eigenvalues of the stress matrix or principal stresses
    '''
    I1 = np.sum(e)
    I2 = e[0]*e[1]+e[1]*e[2]+e[0]*e[2]
    I3 = e[0]*e[1]*e[2]
    return I1, I2, I3

def meanDeviator(stressM,e):                    
    '''Compute the mean stress, mean stress tensor, deviatoric stress 
    tensor.
    input stressM: given stress (or any) matrix with 9 components
    Input e: eigenvalues of the stress matrix or principal stresse
    Input stressM: The stress matrix
    '''
    meanStress = np.mean(e)
    meanMatrix = np.diag([meanStress,meanStress,meanStress])
    deviatorM = stressM - meanMatrix
    return meanStress, meanMatrix, deviatorM

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    ''' provided by ChatGPT'''
    diff = abs(sp.N(a) - sp.N(b))
    return diff <= max(rel_tol * max(abs(sp.N(a)), abs(sp.N(b))), abs_tol)

def printAn(A, n, p_text="A"):
    '''print randomly selected n elements in array A
    '''
    if A.ndim == 1:
        rows = A.shape
        cols =1
    else:
        rows, cols = A.shape
        
    random_idx = np.random.randint(0, rows*cols, size=n)
    random_ele = A.flatten()[random_idx]
    
    print(f'Randomly selected {n} elements: {p_text}:\n{random_ele}')

def printc(c, n_d=4):
    '''print a complex number c with n_d number of significant digits 
       after the decimal place for both real and imaginary parts.
    Inputs: 
       c  : the complex number 
       n_d: number of digits to keep after the decimal place.
    '''
    print(f"{c.real:.{str(n_d)}f} + {c.imag:.{str(n_d)}f}j")

def permuP(perm, module='Numpy'):
    '''Create a permutation matrix P for a given permutation
    Input:   perm: list, given permutations needed.
    module:  if Sympy, P will be Sympy Matrix 
             else, P will be a numpy array
    Return:  P, the corresponding permutation matrix generated. 
    Example: [0, 1, 2]       # original order
              ⇓  ⇓  ⇓         
    perm  =  [2, 0, 1]       # the given permuted order
    '''
    n = len(perm)                              # dimension of the matrix
    if module == 'Sympy':
        I = sp.eye(n)                                  # identity matrix 
        P = sp.zeros(n)              # initialize the permutation matrix
    else: 
        I = np.eye(n)                      
        P = np.zeros((n,n))            

#    for j in range(n):
#        P[:, perm[j]] = I[:,j]                                 # permute 

    P = I[perm,:]
        
    return P                  

def permuP_sp(perm):
    '''Create a permutation matrix P for a given permutation
    Input  : perm: list, given permutations needed.
    Return : P, Sympy Matrix, the corresponding permutation matrix 
             to be generated. 
    Example: [0, 1, 2]       # original order
              ⇓  ⇓  ⇓         
    perm   = [2, 0, 1]       # the given permuted order
    '''
    n = len(perm)                              # dimension of the matrix
    I = sp.eye(n)                                     # identity matrix 
    P = sp.zeros(n)                  # initialize the permutation matrix 

    for j in range(n):
        P[:, perm[j]] = I[:,j]                                # permute 
        
    return P                                                            

def newton_iter(f, df, guess_roots, limit=1000, atol=1e-6):  
    ''' Finds the roots of a function via the Newton iteration.
        inputs: f--the given function;
                df--the derivative of f; 
                guess_roots--possible locations of the roots;
                limit: max number of interations;
                atol:  tolerance to control convergence;
        return: the roots found.'''  
    i = 0;   roots = np.array([])
    for r in guess_roots:
        while lg.norm(f(r)) > atol and i < limit:
            r = r - f(r)/df(r)
            i += 1
        roots = np.append(roots, r)
        
    if lg.norm(f(r)) > atol: 
        print(f'Not yet converged:{atol}!!!')
    else:
        print(f'Converged at {i}th iteration to {atol}')
        
    return roots

def HermiteR(n, x):   
    '''Generate the Laguerre polynomials using the recursion 
    formula, degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return 2*x
    else:  
        Her = 2*x*HermiteR(n-1,x)-2*(n-1)*HermiteR(n-2,x) 
    return Her 

def LegendreR(n, x):   
    '''Generate the Legendre polynomials using the Bonnet’s recursion 
    formula, degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return x
    else:
        leg = ((2*n-1)*x*LegendreR(n-1,x)-(n-1)*LegendreR(n-2,x))/n 
        return leg.expand() 


def PolyAprox(f_true, Ns, x_nodes, Pfit, Px):
    '''To plot the curve fitted using a set of basis functions, and 
    with the true function to be approximated.
    
    f_true:  the true function
    Pfit:    numpy polynomial fitting object for fitting
    Px:      numpy polynomial object 
    Ns:      list, degrees of approximation
    x_nodes: array like, nodes on x-axis used for fitting
    '''
    fx_nodes = [f_true(xi) for xi in x_nodes]  # true f vals for fitting

    for i in range(len(Ns)):
        P_coef = Pfit(x_nodes, fx_nodes, Ns[i])       # fit f for coeffs
        P_ap_x   = Px(P_coef)         # compute the fitted function of x

    return P_ap_x 


def plot_PolyAprox(f_true, Ns, x_nodes, X, Pfit, Px, P_type='Chebyshev'):
    '''To plot the curve fitted by a set of basis functions, to gether 
    with the original function to be approximated.
    
    f_true:  the true function
    Pfit:    numpy polynomial fitting object for fitting
    Px:      numpy polynomial object 
    Ns:      list, degrees of approximation
    x_nodes: array like, nodes on x-axis used for fitting
    X:       array like, points on x-axis for plots
    P_type:  string, tpye of the basis polynomial used, for file name. 
    '''
    fX = [f_true(xi) for xi in X]              # ture function for plots

    fx_nodes = [f_true(xi) for xi in x_nodes]  # true f vals for fitting
    
    plt.rcParams.update({'font.size': 6})
    fig_s = plt.figure(figsize=(6,5))   

    for i in range(len(Ns)):
        ax = fig_s.add_subplot(2,2,i+1)
        P_coef = Pfit(x_nodes, fx_nodes, Ns[i])       # fit f for coeffs
        P_ap_x = Px(P_coef)                          # fitted polynomial
        P_apX   = P_ap_x(X)            # Compute the fitted results at X
                                                          # for plotting
        ax.plot(X,P_apX,'mo',ms=2,label=P_type+' deg.'+str(Ns[i])+
                f', rmse: {sqrt(np.mean((P_apX-fX)**2)):.4e}')
        ax.plot(X, fX, c='k', label='Original Function')
        if i==2 or i==3: ax.set_xlabel('x')
        if i==0 or i==2: ax.set_ylabel('$f(x)$')
        ax.legend()        #loc='center right', bbox_to_anchor=(1, 0.5)) 
        
    plt.savefig('images/'+P_type+'.png',dpi=500,bbox_inches='tight')
    plt.show() 
    
    return P_ap_x 


def cheby_T(n, x):   
    '''Generate the first kind Chebyshev polynomials of degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return x
    else:        return (2*x*cheby_T(n-1,x)-cheby_T(n-2,x)).expand()

def cheby_U(n, x):   
    '''Generate the 2nd kind Chebyshev polynomials of degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return 2*x
    else:        return (2*x*cheby_U(n-1,x)-cheby_U(n-2,x)).expand()

def node_shape_f(𝜉):
    '''Linear nodal shape function in natural coordinate 𝜉.
    This shape function has 4 pieces.''' 
    N = np.piecewise(𝜉, [𝜉<-1, (𝜉>=-1)&(𝜉<0),  (𝜉>=0)&(𝜉<=1),  𝜉>1],\
                        [0,    lambda 𝜉: 1+𝜉,  lambda 𝜉: 1-𝜉,   0]) 
    return N

def uniqueX(X, atol=1.e-6): 
    '''Remove elements in X that has close values within atol'''
    u_ndcs=np.unique([np.isclose(X, v, atol=atol).argmax() for v in X])
    return X[u_ndcs]
## Example: 
#X = np.array([1.0, 2.0, 2.1, 2.2, 3.0, 3.05, 3.1, 5.3, 6.0])
#print(uniqueX(X, atol=1.e-1))

def gauss_df(x,mu,sigma):               
    '''define the Gauss function: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)'''
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)

def errors(a, b): 
    '''Computing the MSE-, Min-, Max-Errors between two 1D arrays
       a and b.
    return: mse, min_error, max_error
    '''
    if a.shape != b.shape: 
        print(f'Warning: a.shape != b.shape: {a.shape, b.shape}')
    
    af = a.flatten(); bf = b.flatten()
    mse = mean_squared_error(af, bf)                # Mean squared error
     
    a_norm=(np.linalg.norm(af,np.inf)-np.linalg.norm(af,-np.inf))/2+1e-8
    #rel_error = np.linalg.norm(af - bf)/a_norm/len(af)
    min_error = np.min((af-bf))/a_norm  
    max_error = np.max((af-bf))/a_norm  
    return mse, min_error, max_error

def LagrangeP(x, xi):
    '''Create shape function matrix N(x)=[l_0(x), l_1(x),...l_degree(x)] 
    in as list for a 1D interval (element) [a=x_0, x_1, ...,  b=x_n], 
    using the Lagrange interpolator.
    '''
    N = []                       # l_0(x) is a shape function N(x)
    for i in range(len(xi)):
        nodes = [n for n in range(len(xi))]
        nodes.pop(i)
        N.append(np.prod([(x-xi[k])/(xi[i]-xi[k]) for k in nodes]))
        
    return N


def LaguerreR(n, x):   
    '''Generate the Laguerre polynomials using the recursion 
    formula, degree (n-1) '''
    if   n == 0: return sp.S.One
    elif n == 1: return -x+1
    else:
        Lag = ((2*n-1-x)*LaguerreR(n-1,x)-(n-1)*LaguerreR(n-2,x))/n 
        return Lag.expand()


def pad_last(arr, c_inc, maxlen):
    '''Extend the given arr to with maxlen columns, by repeating the 
    last row in the array. The c_inc column (usually the time) is set 
    to increase with the same rate of the last two elments in the column
    of the orginal arr.
    Return: arr_expanded.
    '''
    row_add = maxlen - arr.shape[0]      # number of rows to add to arr
    if arr[-2, c_inc] == 0:
        rate = 0
    else:
        rate = (arr[-1, c_inc] - arr[-2, c_inc]) / arr[-2, c_inc]

    # repeat the last row to expand the arr to shape (n+row_add,m)
    arr_add = np.repeat(arr[-1:, :], row_add, axis=0)

    # gradually increase the values in the 1st column of  extended array
    for i in range(row_add):
        arr_add[i,c_inc] = arr[-1,c_inc] + (i+1)*rate*arr[-1,c_inc]

    # concatenate the original array with the expanded array
    arr_expanded = np.concatenate((arr, arr_add), axis=0)
    return arr_expanded

## Example: 
# n=9; m=5; row_add=2; c_inc = 0 
# arr = np.arange(n*m).reshape(n, m)   # create a numpy array
# print(f'Original array: {arr.shape} \n{arr}') 
# arr_expanded = pad_last(arr, c_inc, n+row_add)
# print(f'Extended array: {arr_expanded.shape} \n{arr_expanded}') 

def array_to_list(dictionary):
    """ Recursively convert all NumPy arrays in a dictionary to lists.
        Suggested by ChatGPT. 
    """
    for k, v in dictionary.items():
        if isinstance(v, np.ndarray): dictionary[k] = v.tolist()
        elif     isinstance(v, dict): array_to_list(v)


def flattenL(nested_list):
    '''flatten lists with multiple depth, recursively. 
       Suggested by ChatGPT'''
    flat_list = []
    for item in nested_list:
        if isinstance(item, list): flat_list.extend(flatten_list(item))
        else:                      flat_list.append(item)
    return flat_list

def roundS(expr, n_d):
    '''Usage: roundS(expr, n_d), where n_d: number of digits to keep.
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    return expr.xreplace({n.evalf():round(n,n_d) 
                          for n in expr.atoms(sp.Number)})

def solver1D2(E, A, bf, l, bc0, bcl, key='disp'):
    
    '''Solves mechanics prolems for 1D bars subjected to body 
    force bf: u,xx=-bf(x)/EA, with BCs: u(0)=bc0; u(l)=bcl or fR=bcl.
    Input: bf, body force; l, the length of the bar; 
           bcl, condition at x=l; 
           key="disp", displacement condition at a boundary;
           otherwise a force condition
    Return: u, E*u_x, u_xx ''' 
    
    # Integrate twice to obtain the general solutions in u_x and u: 
    u_x = sp.integrate(-bf/E/A,(x))+ c0  #c0: integration constant
    u   = sp.integrate(u_x,(x))+ c1      #c1: integration constant
    
    # Solve for the integration constants: 
    if key == "disp":
        consts=sp.solve([u.subs(x,0)-bc0, u.subs(x,l)-bcl],  [c0, c1])
    else:
        consts=sp.solve([u.subs(x,0)-bc0,u_x.subs(x,l)-bcl/E/A],[c0,c1])
        
    # Substitute the constants back to the general solutions, to obtain
    # solutions that satisfy the BCs:
    u   =   u.subs({c0:consts[c0],c1:consts[c1]})
    u   =   u.expand().simplify()
    u_x = u_x.subs({c0:consts[c0],c1:consts[c1]})
    u_x = u_x.expand().simplify()
    u_xx = sp.diff(u_x,x)
    
    print(f'output from solver1D2(): u, 𝜎, u_xx')
    return u.expand(), (E*u_x).expand(), u_xx.expand()

def maxminS(f, x, title="value"):
    ndg = 8           # number of digits
    df = sp.diff(f, x)
    ddf = sp.diff(df, x)
    df0_points = sp.solve(df, x)     #find stationary points
    df0s = [roundS(point.evalf(),ndg) for point in df0_points]
    print(f"Stationary points for {title}: {df0s}")

    for point in df0s:
        fv = roundS((f.subs(x,point)).evalf(), ndg)
        ddfv = ddf.subs(x, point)
        if ddfv < 0:
            print(f"At x={point}, local maximum {title}={fv}")
        elif ddfv > 0:
            print(f"At x={point}, local minimum {title}={fv}")
        else:
            print(f"At x={point}, max or min {title}={fv}")

    df0s.append(0)        # Add in points on the boundaries
    df0s.append(1)
    fs_df0 = [f.subs(x, point).evalf() for point in df0s]
    f_max = roundS(max(fs_df0), ndg)
    x_max=[pnt for pnt in df0s if ma.isclose(f.subs(x,pnt),max(fs_df0))]

    print(f"\nAt x={x_max}, Max {title}={f_max}\n")

def plot2curveS(u, xL=0., xR=1., title="f_title"):
    '''Print out maximum values and loctions, as well as stationary 
    points, and the values at the stationary points, and boundaries'''
    x = sp.symbols('x')
    dx = 0.01; dxr = dx*10       # x-step
    xi = np.arange(xL, xR+dx, dx)
    uf = sp.lambdify((x), u[0], 'numpy')   #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) != np.ndarray:       #in case, uf is a constant
        #type(yi) == int or type(yi) == float: # or len(yi)==1:  
        xi = np.arange(xL, xR+dxr, dxr)
        yi = float(yi)*np.ones_like(xi)

    fig, ax1 = plt.subplots(figsize=(5.,1.), dpi=300)
    fs = 8          # fontsize
    color = 'black'
    ax1.set_xlabel('location x', fontsize=fs)
    ax1.set_ylabel(title[0], color=color, fontsize=fs)
    ax1.plot(xi, yi, color=color)
    ax1.grid(color='r',ls=':',lw=.3, which='both') # Use both tick 
    ax1.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fs)

    vmax = yi[yi.argmax()]
    max_l = np.argwhere(yi == vmax)
    ax1.plot(xi[max_l],  yi[max_l], 'r*', markersize=4)
    print(f'Maximum {title[0]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
    
    uf = sp.lambdify((x), u[1], 'numpy') #convert Sympy f to numpy f 
    xi = np.arange(xL, xR+dx, dx)
    yi2 = uf(xi)
    if type(yi2) != np.ndarray: # or len(yi2) == 1:
        xi = np.arange(xL, xR+dxr, dxr)
        yi2 = float(yi2)*np.ones_like(xi)

    m1, m2, m3 = np.partition(abs(yi2), 2)[0:3]
    msl=[np.where(abs(yi2)==m1)[0][0],np.where(abs(yi2)==m2)[0][0],
             np.where(abs(yi2)==m3)[0][0]]
    
    vmax = yi2[yi2.argmax()]
    max_l = np.argwhere(yi2 == vmax)
    print(f'Maximum {title[1]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
        
    if abs(xi[msl[2]]-xi[msl[1]])<2*dx:
        if abs(yi2[msl[2]]-0.)<abs(yi2[msl[1]]-0.): msl.pop(1)
        else: msl.pop(2) 
    if len(msl) > 2:
        if abs(xi[msl[2]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[2]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(2)  
    if len(msl) > 1:
        if abs(xi[msl[1]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[1]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(1)
         
    ax2 = ax1.twinx()  # instantiate second axes sharing the same x-axis

    color = 'blue'
    ax2.set_ylabel(title[1], color=color, fontsize=fs) 
    ax2.plot(xi, yi2, color=color)
    ax2.plot(xi[max_l], yi2[max_l], 'r*', markersize=4)
    ax2.plot(xi[msl], yi2[msl], 'r*', markersize=4)
    ax1.plot(xi[msl],  yi[msl], 'r*', markersize=4)
    ax2.plot(xi[0], yi2[0], 'ro', markersize=2)
    ax1.plot(xi[0],  yi[0], 'ro', markersize=2)
    ax2.plot(xi[-1], yi2[-1], 'ro', markersize=2)
    ax1.plot(xi[-1],  yi[-1], 'ro', markersize=2)
    ax2.grid(color='r',ls=':',lw=.5, which='both') # Use both tick  
    ax2.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fs)
    np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
    print(f'Extreme {title[0]} values={yi[msl]},\n    at x={xi[msl]}')
    print(f'Critical {title[1]} values={yi2[msl]},\n    at x={xi[msl]}')
    print(f'{title[0]} values at boundary ={yi[0], yi[-1]}')
    print(f'{title[1]} values at boundary ={yi2[0], yi2[-1]}\n')

def plot2curveS0(u, xL, xR, title="f_title"):
    dx = 0.01
    xi = np.arange(0., 1.+dx, dx)
    uf = sp.lambdify((x), u[0], 'numpy') #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) == int: # or len(yi) == 1:
        yi = float(yi)*np.ones_like(xi)

    fig, ax1 = plt.subplots(figsize=(5.,1.), dpi=300)
    fs = 8     # fontsize
    color = 'black'
    ax1.set_xlabel('location x', fontsize=fs)
    ax1.set_ylabel(title[0], color=color, fontsize=fs)
    ax1.plot(xi, yi, color=color)
    ax1.grid(color='r',ls=':',lw=.3, which='both') # Use both tick 
    ax1.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fs)

    uf = sp.lambdify((x), u[1], 'numpy') #convert Sympy f to numpy f 
    yi2 = uf(xi)
    if type(yi2) == int: # or len(yi2) == 1:
        yi2 = float(yi2)*np.ones_like(xi)
        
    m1, m2, m3 = np.partition(abs(yi2), 2)[0:3]
    msl=[np.where(abs(yi2)==m1)[0][0],np.where(abs(yi2)==m2)[0][0],
             np.where(abs(yi2)==m3)[0][0]]
    
    vmax = yi[yi.argmax()]
    max_l = np.argwhere(yi == vmax)
    ax1.plot(xi[max_l],  yi[max_l], 'r*', markersize=4)
    print(f'Maximum {title[0]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
    vmax = yi2[yi2.argmax()]
    max_l = np.argwhere(yi2 == vmax)
    print(f'Maximum {title[1]} value={vmax:.3e}, at x={xi[max_l][0][0]}')
        
    if abs(xi[msl[2]]-xi[msl[1]])<2*dx:
        if abs(yi2[msl[2]]-0.)<abs(yi2[msl[1]]-0.): msl.pop(1)
        else: msl.pop(2) 
    if len(msl) > 2:
        if abs(xi[msl[2]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[2]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(2)  
    if len(msl) > 1:
        if abs(xi[msl[1]]-xi[msl[0]])<2*dx:
            if abs(yi2[msl[1]]-0.)<abs(yi2[msl[0]]-0.): msl.pop(0)
            else: msl.pop(1)
         
    ax2 = ax1.twinx()  # instantiate second axes sharing the same x-axis

    color = 'blue'
    ax2.set_ylabel(title[1], color=color, fontsize=fs) 
    ax2.plot(xi, yi2, color=color)
    ax2.plot(xi[max_l], yi2[max_l], 'r*', markersize=4)
    ax2.plot(xi[msl], yi2[msl], 'r*', markersize=4)
    ax1.plot(xi[msl],  yi[msl], 'r*', markersize=4)
    ax2.plot(xi[0], yi2[0], 'ro', markersize=2)
    ax1.plot(xi[0],  yi[0], 'ro', markersize=2)
    ax2.plot(xi[-1], yi2[-1], 'ro', markersize=2)
    ax1.plot(xi[-1],  yi[-1], 'ro', markersize=2)
    ax2.grid(color='r',ls=':',lw=.5, which='both') # Use both tick  
    ax2.tick_params(axis='x', labelcolor=color, labelsize=fs)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fs)
    np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
    print(f'Extreme {title[0]} values={yi[msl]},\n    at x={xi[msl]}')
    print(f'Critical {title[1]} values={yi2[msl]},\n    at x={xi[msl]}')
    print(f'{title[0]} values at boundary ={yi[0], yi[-1]}')
    print(f'{title[1]} values at boundary ={yi2[0], yi2[-1]}\n')
    #Vmax = max(max(yi[msl]),yi[0], yi[-1])
    #V_xmax = max(max(yi2[msl]),yi2[0], yi2[-1])
    #print(f'Maximum {title[0]} values={Vmax}')
    #print(f'Maximum {title[1]} values={V_xmax}\n') 

def plotbeam2by2(u, axis,iax=0, jax=0, title="f_title"):
    '''Plots 4 curves in 2 by 2 arrangement. Example: 
    title = [["deflection", "rotation"], ["moment", "shear force"]]
    fig, axis = plt.subplots(2,2,figsize=(10.,5), dpi=300)
    for i in range(len(title)):
        for j in range(len(title[0])):
            plotbeam(vx, axis, iax=i, jax=j, title=title[i][j]) 
    fig.tight_layout()
    plt.savefig('images/beam_cq.png', dpi=500)
    plt.show()'''
    spacing = 0.5       # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    xi = np.arange(0., 1., .01)
    uf = lambdify((x), u[iax*2+jax], 'numpy') #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) == int or len(yi) == 1:
        yi = float(yi)*np.ones_like(xi)
    axis[iax,jax].plot(xi, yi, label=title)
    axis[iax,jax].set_axisbelow(True)
    axis[iax,jax].set_xlabel('x')
    axis[iax,jax].yaxis.set_minor_locator(minorLocator) # Set minor tick 
    axis[iax,jax].xaxis.set_minor_locator(minorLocator)
    axis[iax,jax].grid(color='r',ls=':',lw=.5, which='both')    
    #axis[iax,jax].set_title(title)
    axis[iax,jax].legend()

def plotbar0(u, axis,iax=0, title="f_title"):
    '''Plots 2 curves. Example:
    title = ["Displacement", "Strain"]
    fig, axis = plt.subplots(1,2,figsize=(10.,1.8), dpi=300)
    for i in range(len(title)):
        plotbar(ux, axis,iax=i, title=title[i]
    plt.show())''' 
    
    spacing = 0.5       # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    xi = np.arange(0., 1., .01)
    uf = lambdify((x), u[iax], 'numpy') #convert Sympy f to numpy f 
    yi = uf(xi)
    if type(yi) == int or len(yi) == 1:
        yi = float(yi)*np.ones_like(xi)
    axis[iax].plot(xi, yi)
    axis[iax].set_axisbelow(True)
    axis[iax].set_xlabel('x')
    axis[iax].yaxis.set_minor_locator(minorLocator) # Set minor tick 
    axis[iax].xaxis.set_minor_locator(minorLocator)
    axis[iax].grid(color='r',ls=':',lw=.5, which='both') # Use both tick  
    axis[iax].set_title(title)

def transferM(theta, about = 'z'):
    '''Create a transformation matrix for coordinate transformation (numpy)\
    Input theta: rotation angle in degree \
          about: the axis of the rotation is about \
    Return: numpy array of transformation matrix of shape (3,3)'''
    from scipy.stats import ortho_group
    
    n = 3        # 3-dimensonal problem
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    #T = np.zeros((n,n))
    
    if about == 'z':
        # rotates about z by theta 
        T = np.array([[ c, s, 0.],
                      [-s, c, 0.],
                      [0.,0., 1.]]) 
    elif about == 'y':
        # rotates about y by theta 
        T = np.array([[ c, 0.,-s],
                      [0., 1.,0.],
                      [s, 0., c]])  
    elif about == 'x':
        # rotates about x by theta 
        T = np.array([[ 1.,0., 0.],
                      [ 0., c, s],
                      [ 0.,-s, c]])
    else: # randomly generated unitary matrix->transformation matrix, no theta
        T = ortho_group.rvs(dim=n)          # Generate a random matrix
        T[2,:] = np.cross(T[0,:], T[1,:])   # Enforce the righ-hand rule
        
    return T, about


def rotationM(theta, about = 'z'):
    '''Create a rotation matrix for coordinate transformation (numpy)\
    Input theta: rotation angle \
          about: the axis of the rotation is about \
    Return: numpy array of rotation matrix of shape (3,3)'''
    from scipy.stats import ortho_group
    
    n = 3        # 3-dimensonal problem
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    #T = np.zeros((n,n))
    
    if about == 'z':
        # rotates about z by theta 
        T = np.array([[ c, s, 0.],
                      [-s, c, 0.],
                      [0.,0., 1.]]) 
    elif about == 'y':
        # rotates about y by theta 
        T = np.array([[ c, 0., s],
                      [0., 1.,0.],
                      [-s, 0., c]])  
    elif about == 'x':
        # rotates about x by theta 
        T = np.array([[ 1.,0., 0.],
                      [ 0., c, s],
                      [ 0.,-s, c]])
    else: # randomly generated unitary matrix->rotation matrix, no theta
        T = ortho_group.rvs(dim=n)          # Generate a random matrix
        T[2,:] = np.cross(T[0,:], T[1,:])   # Enforce the righ-hand rule
        
    return T.T, about   # R=T.T # Rotation matrix

def rotationMs():
    '''define the transformation matrix T for symbolic computation'''

    a11, a12, a13 = sp.symbols("a11, a12, a13")
    a21, a22, a23 = sp.symbols("a21, a22, a23")
    a31, a32, a33 = sp.symbols("a31, a32, a33")
    

    T = sp.MatrixSymbol('T', 3, 3)
    T = sp.Matrix([[a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33]])
    return T.T  # Rotation matrix R=T.T      

def transferMs():
    '''define the transformation matrix T for symbolic computation'''

    a11, a12, a13 = sp.symbols("a11, a12, a13")
    a21, a22, a23 = sp.symbols("a21, a22, a23")
    a31, a32, a33 = sp.symbols("a31, a32, a33")
    
    T = sp.MatrixSymbol('T', 3, 3)
    T = sp.Matrix([[a11, a12, a13],
                [a21, a22, a23],
                [a31, a32, a33]])
    return T

def transferMts(𝜃):
    '''define the transformation matrix T for symbolic computation
       for given rotation angle 𝜃 with respect to z.'''

    T = sp.MatrixSymbol('T', 3, 3)
    T = sp.Matrix([[sp.cos(𝜃),  sp.sin(𝜃), 0],
                   [-sp.sin(𝜃), sp.cos(𝜃), 0],
                   [0,             0,      1]])
    return T

def transf_YPRs(𝜃, about = 'z'):
    '''Create a transformation matrix for coordinate transformation\
    Input 𝜃: rotation angle, in Sympy \
          about: the axis of the rotation is about \
    Return: Sympy matrix of transformation matrix of shape (3,3)
    '''
    c, s = sp.cos(𝜃), sp.sin(𝜃)
    
    if about == 'z':
        # Yaw: rotates about z by 𝜃 
        T = sp.Matrix([[ c, s, 0],
                       [-s, c, 0],
                       [ 0, 0, 1]]) 
    elif about == 'y':
        # Pitch: rotates about y by 𝜃 
        T = sp.Matrix([[ c, 0,-s],
                       [ 0, 1, 0],
                       [ s, 0, c]])  
    elif about == 'x':
        # roll: rotates about x by 𝜃 
        T = sp.Matrix([[ 1, 0, 0],
                       [ 0, c, s],
                       [ 0,-s, c]])
    else: # randomly generated unitary matrix as transformation matrix:
        print("Rotation axis must be given: z, y, or x")
        
    return T, about

def Tensor2_transfer(T,S):
    '''Sybolic coordinate transformation for 2nd order tensors
    '''
    S = np.tensordot(T, S, axes=([1],[0]))
    S = np.tensordot(S, T, axes=([1],[1]))
    return S

def mgs_QR(A):
    '''QR decomposition via modified Gram-Schmidt orthogonalization.
    This is a minimalistic version for principle demonstration. 
    A(m, n): Inputs, an numpy array 
    Q(m, n) and R(n, n): Outputs, numpy arrays 
    Usage: Q, R = mgs_QR(A)''' 
    
    m, n = A.shape               # get the dimensions 
    dtypeA = A.dtype             # get the data type of A
    v = A.copy().astype(dtypeA)    
    Q = np.zeros((m, n),dtype = dtypeA)      # initialization
    R = np.zeros((n, n),dtype = dtypeA)
    
    for i in range(n):           # n iterations needed in GS via
        R[i,i] = lg.norm(v[:,i]) # recursive improvements
        Q[:,i] = v[:,i]/R[i,i]   # normalization
        for j in range(i+1, n):  # 𝐏⊥𝑢 on each component just once
            R[i,j] = np.dot(Q[:,i], v[:,j])  
            v[:,j] -= R[i,j]*Q[:,i]

    return Q, R


def A_exam(A, atol = 1.e-10, n_dgt=4):
    '''Function to examing the properties of a given matrix A'''
    from sympy import Matrix 
    np.set_printoptions(precision=n_dgt,suppress=True)
    print(f'The original matrix:\n{A}')
    B = Matrix(A) 
    A_rank = B.rank()      
    print(f'Rank of the original matrix:{A_rank}')  
    
    e, V = lg.eig(A)
    print(f'Eigenvalues e:{e}')
    print(f'Eigenvectors V:\n{V}')
    print(f'Condition number of V={lg.cond(V)}')
    B = Matrix(V)
    V_rank = B.rank(iszerofunc= lambda x: abs(x)<atol) 
    print(f'Rank of eigenvector matrix V ={V_rank}')  
    print(f'Determinant of the original matrix ={np.prod(e)}')
    return e, V, A_rank, V_rank 

def AI_exam(A, e, ie, atol = 1.e-10, n_dgt=4):
    ''' Function to examing the properties of [A-lambda*I]'''
    from sympy import Matrix 
    np.set_printoptions(precision=n_dgt,suppress=True)
    AI = A-e[ie]*np.eye(len(e))
    print(f'Matrxi [A-lambda*I]:\n{AI}')
    B = Matrix(AI)
    nullity = B.shape[1] - B.rank(iszerofunc= lambda x: abs(x)<atol) 
    print(f'Nullity of [A-("+str(e[ie])+")*I]={nullity}')  

    # If want to further examine the nullpace: 
    if nullity == 0:               
        print(f'The nullpace has only the zero-vector!')
    else: 
        NullSpace=B.nullspace(iszerofunc= lambda x: abs(x)<atol) # list
        print(f'Null Space:{NullSpace[0]}')         # Select the Matrix 
        print(f'Is Ax=0? {B@NullSpace[0]}')     # check if in nullspace

def A_diag(A, e, V, n_dgt=4):
    '''Produce diagonalized A, given e and V can then check the \
    correctness'''
    np.set_printoptions(precision=n_dgt,suppress=True)
    print(f'Inverse of the eigvenvector matrix:\n{lg.inv(V)}') 
    print(f'Original matrix:\n{A}')
    E = np.diag(e) # matrix Lambda with eigenvalues at diagonal
    print(f'Diagonalized, with eigenvalues at diagonal:\n{E}')
    A_recovered = V@E@lg.inv(V)
    print(f'Reconstructed A via EVD:\n{A_recovered}')

def eig_diag(A, atol = 1.e-10, n_dgt=4):
    '''Perform EVD for matrix A, and check the correctness'''
    np.set_printoptions(precision=n_dgt,suppress=True)
    print(f'Original matrix:\n{A}',f'Rank={lg.matrix_rank(A)}')
    e, V = lg.eig(A)
    V_inv = lg.inv(V)               
    print(f'Eigenvalues, diagonalized matrix:\n{np.diag(e)}')
    print(f'The eigenvector matrix:\n{V}',f'Cond={lg.cond(V)}')
    print(f'Inverse of the eigenvector matrix:\n{V_inv}')
    Ae = V@np.diag(e)@V_inv  # Reconstructed matrix
    print(f'Are eigenvalues correct?\n{np.isclose(Ae,A,atol=atol)}')
    return e, V, V_inv 

def printE(e, n_dgt=4):
    '''Print out eigenvalues with multiplicities for sympy.dict
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    np.set_printoptions(precision=n_dgt,suppress=True)
    fstr = '{: 0.'+str(n_dgt)+'f}'
    np.set_printoptions(formatter={'float': fstr.format})
    #print(f'Eigenvalues and multiplicities:{e}')
    #ea = np.array(list(e.items()))
    ei=np.array(list(e.keys()),dtype=float) #dict->list->numpy array
    me=np.array(list(e.values()), dtype=int)
    print(f'Eigenvalues={ei}, m={me}')
    n_e = np.sum(me, axis=0)             # number of eigenvalues
    print(f'Totoal number of eigenvalues ={n_e}') 
    #printM(sp.Matrix(ei).T)
    return ei, me, n_e                   #in numpy array

def printM0(A, p_text='Matrix A', n_dgt=4):
    '''Print out a sympy.Matrix A in latex form
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    from sympy import init_printing, Float, Number
    init_printing(wrap_line=True,scale=0.85)
    print(p_text)
    return A.xreplace({n.evalf() : n if type(n)==int else 
           Float(n, n_dgt) for n in A.atoms(Number)}) 

def printM(A, *text, n_dgt=4):
    '''Print out a sympy.Matrix A in latex form
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    from sympy import init_printing, Float, Number
    init_printing(wrap_line=True,scale=0.85)
    print(text[0])
    return A.xreplace({n.evalf() : n if type(n)==int else 
           Float(n, n_dgt) for n in A.atoms(Number)}) 

def printCM(y_test,y_predict):
    print(f"Classification Report: \n" 
          f"{classification_report(y_test,y_predict,labels=list(range(10)))}")

    # Compute the Modified confusion matrix: 
    digids = [i for i in range(11)] #; digids[-1] = '/'
    digids = np.array(digids)[:,np.newaxis]
    allLabels = [str(i) for i in range(10)] #print(y_test, y_predict, allLabels)
    c_matrix = confusion_matrix(y_test,y_predict, labels=allLabels)
    supports = np.sum(c_matrix, axis=1)
    TP_FN = np.sum(c_matrix, axis=1)  # supports
    c_matrix = np.hstack((c_matrix, TP_FN[:,np.newaxis]))
    TP_FP = np.sum(c_matrix, axis=0)
    c_matrix = np.vstack((c_matrix, TP_FP[np.newaxis,:]))
    c_matrix = np.hstack((digids, c_matrix))
    #print(f"Confusion matrix:\n{c_matrix}"); print(f"Supports:\n{supports}")
    print(f"Extended confusion matrix:\n{c_matrix}")

def eval_svm(y_test, y_pred, warning = False):
    '''Evaluate of the performance of a model using Sklearn.'''
    # Overall accuracy of the prediction
    print(f"Overall accuracy = {accuracy_score(y_test, y_pred)*100} %")

    # Detailed classification results and performance report
    if warning: warnings.filterwarnings("ignore")

    from sklearn.metrics import classification_report, confusion_matrix
    print(f"Classification Report: \n" 
          f"{classification_report(y_test,y_pred,labels=list(range(10)))}")

    # Compute the Modified confusion matrix: 
    digids = [i for i in range(11)] #; digids[-1] = '/'
    digids = np.array(digids)[:,np.newaxis]
    allLabels = [str(i) for i in range(10)] #print(y_test, y_pred)
    
    c_matrix = confusion_matrix(y_test,y_pred, labels=allLabels)
    supports = np.sum(c_matrix, axis=1)
    
    TP_FN = np.sum(c_matrix, axis=1)  # supports
    c_matrix = np.hstack((c_matrix, TP_FN[:,np.newaxis]))
    TP_FP = np.sum(c_matrix, axis=0)
    
    c_matrix = np.vstack((c_matrix, TP_FP[np.newaxis,:]))
    c_matrix = np.hstack((digids, c_matrix))
 
    print(f"Modified confusion matrix:\n{c_matrix}")

def printl(list_num, *arg, n_d=4):
    '''print given list, list_num, with numerical numbers in format of .n_df'''
    str_list = [f"{num:.{str(n_d)}f}" for num in list_num]
    str_list = [float(num) for num in str_list]
    print(*arg, str_list)

def printt(tuple_num, *arg):
    '''print given tuple_num, with numerical numbers in format of .4f'''
    str_tuple = [f"{num:.4f}" for num in tuple_num]
    str_tuple = [float(num) for num in str_tuple]
    print(*arg, str_tuple)

def printA(A, p_text='Matrix A:', n_dgt=4):
    '''Print out a numpy.Matrix A in latex form
    Ref: https://stackoverflow.com/questions/48491577/
    printing-the-output-rounded-to-3-decimals-in-sympy'''
    from sympy import init_printing, Float, Number, Matrix
    np.set_printoptions(precision=n_dgt,suppress=True)
    init_printing(wrap_line=True,scale=0.85)
    if p_text != "None":
        print(p_text)
    return Matrix(A).xreplace({n.evalf() : n if type(n)==int else 
           Float(n, n_dgt) for n in Matrix(A).atoms(Number)}) 

def A_eig(A, p_Av=True, atol=1e-12, n_dgt=4):
    np.set_printoptions(precision=n_dgt, suppress=True)
    print(f'Original matrix:\n{A}')
    e, V = lg.eig(A)      # Invoke np.linalg.eig to compute eigenvalues
    print(f'Eigenvalues of the matrix:{e}')
    print(f'Eigenvectors of the matrix:\n{V}')
    if p_Av: 
        print(f'Av:\n{A@V}')   # or use A.dot(v)
        print(f'Lambda*v:\n{V@np.diag(e)}')            # to get e_i*v_I
        
    print(f'Is Av=Lambda*v?\n{np.isclose(A@V,V@np.diag(e),atol=1e-12)}') 
    return e, V

def printEig(e, V):
    '''Print out eigenvalues and its corresponding eigenvectors'''
    np.set_printoptions(precision=3)
    for eig_val, eig_vec in zip(e, V.T):
        print(f'e={eig_val:0.3f}, ev={eig_vec}')

def sp_vcts(V_sp):
    '''Get eigenvectors from sympy eigenvectors V_sp in a dict,\
       and its geometric multiplicity nv_np'''
    V_list = []
    nv = []
    for tup in V_sp:
        nv.append(len(tup[2]))
        for v in tup[2]:
            V_list.append(list(v))
    V_np = np.array(V_list)
    nv_np =np.array(nv)
    return V_np.T, nv_np  

def sp_nulls(V_sp):              #in numpy array
    '''Get eigenvectors from sympy matrix V_sp in a list'''
    V_list = []
    for v in V_sp:
        V_list.append(list(v))
    V_np = np.array(V_list)  
    return V_np.T 

def svd_diag(A, atol = 1.e-10):
    '''Perform SVD for arbitrary matrix A and check the correctness'''
    U, s, VT = lg.svd(A)           #SVD
    print(f"Original Matrix:\n{A}, lg rank={lg.matrix_rank(A)}")
    S = np.diag(s)
    r = len(s)                     # rank of the matrix
    print(f"SVD, left-singular vectors (after cut-off):\n{U[:,0:r]}")
    print(f"SVD, Singular values (diagonal):\n{s}")
    print(f"SVD, right-singular vectors (after cut-off):\n{VT[0:r,:]}")
    As = U[:,:r]@S[:r,:r]@VT[:r,:] #Reconstruction of the matrix
    print(f"SVD, U@S@VT:\n{As}")
    print(f"Is U@S@VT the original? \n{np.isclose(A,As,atol=atol)}")
    return U, s, VT

def null_exam0(A):
    '''Examination of numpy array A by performing row-echolen-from.  
    Tested on two singular marices: 
    #A = np.array([[1,-0.2, 0], [0.1, 1,-0.5], [0.1, 1,-0.5]]) and 
    #A=np.array([[1,-0.2,0,0],[0.1,1,-0.5,0],[0.1,1,-0.5,0],[0.1,1,-0.5,0]])
    And one nonsingular matrix:
    #A = np.array([[1, 1], [0, 1]])        # the shear matrix
    #A = np.array([[1,-0.2, 0], [0.1, 1,-0.5], [0.1, 1,-0.8]])
    '''
    from sympy import Matrix
    #np.set_printoptions(precision=3)
    n = np.shape(A)
    M = Matrix(A)
    M_rref = M.rref()
    print("Row echelon form of A with pivot columns:{}".format(M_rref))
    # Basis for null-space (see the textbook by Gilbert Strong)
    r = np.size(M_rref[1])
    print(A,'shape:',n,'\nRank=',r,', or dimension of C(A)')
    nd = n[1] - r
    print('Dimension of the null-space N(A), or nullity =',nd)
    if nd == 0: 
        print('Thus, the null-space has only the zero vector')
    else: 
        N = - M_rref[0][:,r:n[1]+1]
        print('Basis vector for the null-space:\n',N)
        print(n[1],np.shape(N), np.eye(nd))
        N[r:n[1],0:nd] = np.eye(n[1]-r)
        print('Basis vector for the null-space:\n',N)
        print('Transformed basis vector of the null-space by A:\n', A.dot(N),
              ', N is orthogonal to all the row-vecors of A: A@N=0')
        print('N is also orthogonal to the rref of A:\n',M_rref[0]@N)

###########################################################################

def plot_fs(x, y, labels, xlbl, ylbl, *p_name):
    '''plot multiple x-y curves in one figure.
    x: x data, list of 1D numpy array;
    y: y data, list of 1D numpy array 
    labels: labels for these curves
    xlbl: for ax.set_xlabel()
    ylbl: for ax.set_ylabel()
    p_name: string, name of the plot. 
    '''
    colors = ['b', 'r', 'g', 'c', 'm','k','y','w'] 
    plt.rcParams.update({'font.size': 5})         # settings for plots
    fig_s = plt.figure(figsize=(4,2.5))   
    ax = fig_s.add_subplot(1,1,1)
    for i in range(len(y)):
        plt.plot(x[i], y[i], label=labels[i], color=colors[i],lw=0.9) 
    
    ax.grid(c='r', linestyle=':', linewidth=0.5)
    ax.axvline(x=0, c="k", lw=0.6);ax.axhline(y=0, c="k", lw=0.6)
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    ax.legend() #loc='center right', bbox_to_anchor=(1, 0.5)) 
    #plt.title('x-y curves'+p_name+'')

    plt.savefig('images/'+p_name[0]+'.png',dpi=500,bbox_inches='tight')
    plt.show()

def plot_fs0(x, y, labels, xlbl, ylbl, *p_name):
    '''plot multiple x-y curves in one figure.
    x: x data points, 1D numpy array;
    y: y data, list of 1D numpy array 
    labels: labels for these curves
    xlbl: for ax.set_xlabel()
    ylbl: for ax.set_ylabel()
    p_name: string, name of the plot. 
    '''
    colors = ['b', 'r', 'g', 'c', 'm'] 
    plt.rcParams.update({'font.size': 5})         # settings for plots
    fig_s = plt.figure(figsize=(4,2.5))   
    ax = fig_s.add_subplot(1,1,1)
    for i in range(len(y)):
        plt.plot(x, y[i], label=labels[i], color=colors[i]) 
    
    ax.grid(c='r', linestyle=':', linewidth=0.5)
    ax.axvline(x=0, c="k", lw=0.6);ax.axhline(y=0, c="k", lw=0.6)
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    ax.legend() #loc='center right', bbox_to_anchor=(1, 0.5)) 
    #plt.title('x-y curves'+p_name+'')

    plt.savefig('images/'+p_name[0]+'.png',dpi=500,bbox_inches='tight')
    plt.show()

def plot_fij(fig_s, x, y, labels, xlbl, ylbl,fij):
    '''plot multiple x-y curves in multiple figures.
    fig_s: a plt.figure() object
    x: x data points, 1D numpy array
    y: y data, list of 1D numpy array 
    labels: labels for these curves
    xlbl: for ax.set_xlabel()
    ylbl: for ax.set_ylabel()
    fij: figure grids. First 2 idexes gives the grid, 3rd is for 
         figure number. The last one controls whether plot xlabel. 
    '''
    colors = ['b', 'r', 'g', 'c', 'm', 'k'] 

    ax = fig_s.add_subplot(fij[0],fij[1],fij[2])  #(1,1, 1)
    for i in range(len(y)):
        ax.plot(x, y[i], label=labels[i], color=colors[i]) 
    
    ax.grid(c='r', linestyle=':', linewidth=0.5)
    #ax.axvline(x=0, c="k", lw=0.6);ax.axhline(y=0, c="k", lw=0.6)
    if fij[3] == 'xl': ax.set_xlabel(xlbl) 
    if fij[4] == 'yl': ax.set_ylabel(ylbl)
    ax.legend() #loc='center right', bbox_to_anchor=(1, 0.5)) 
    
    #plt.title('x-y curves'+p_name+'')
    #plt.savefig('images/'+p_name[0]+'.png',dpi=500,bbox_inches='tight')
    #plt.show()


def plotfig(axis, iax=0, title="f_title", xylim = 6):
    from matplotlib.ticker import MultipleLocator 
    xl, xr = -xylim, xylim                  # x left, x right for plots
    yf, yc = -xylim, xylim                  # y floor, y ceiling for plots
    spacing = 1        # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    axis[iax].set_axisbelow(True)
    axis[iax].yaxis.set_minor_locator(minorLocator) # Set minor tick locations
    axis[iax].xaxis.set_minor_locator(minorLocator)
    axis[iax].grid(ls=':',lw=.5, which = 'both')              # Use both ticks 
    axis[iax].axis('square')       
    axis[iax].plot(0, 0, 'ro', markersize=2)
    axis[iax].set_xlim([xl,xr])
    axis[iax].set_ylim([yf,yc])
    axis[iax].set_title(title)


def drawArrow(axis,origin,vector,color='k',iax=0,xylim = 6,label='None'):
    aw, lw = 0.015*xylim, 0.09*xylim # 0.07, 0.3      # arrow width, line-width
    if label == 'None':
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw, 
            color=color, fill=False, length_includes_head=True)
        labels = label
    else:
        labels = axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],
                 width=aw,lw=lw, color=color, label=label, fill=False, 
                 length_includes_head=True)
    return labels
    
def drawArrow0(axis,origin,vector,color='k',iax=0,xylim = 6, label='No'):
    aw, lw = 0.015*xylim, 0.09*xylim # 0.07, 0.3      # arrow width, line-width
    if label == 'None':
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw, 
            color=color, fill=False, length_includes_head=True)
    else:
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw,
            color=color, label=label, fill=False, length_includes_head=True)
        #axis[iax].legend()
       
###############################################################################

def plotfig0(axis,iax=0, title="f_title",axislim = 6):
    xl, xr = -axislim, axislim                     # x left, x right for plots
    yf, yc = -axislim, axislim                  # y floor, y ceiling for plots
    spacing = 1        # grid line spacing. 
    minorLocator = MultipleLocator(spacing)
    axis[iax].set_axisbelow(True)
    axis[iax].yaxis.set_minor_locator(minorLocator) # Set minor tick locations
    axis[iax].xaxis.set_minor_locator(minorLocator)
    axis[iax].grid(ls='--',lw=1.2, which = 'both')    # Use both tick locations 
    axis[iax].legend()
    axis[iax].axis('square')
    axis[iax].plot(0, 0, 'ro', markersize=6)
    axis[iax].set_xlim([xl,xr])
    axis[iax].set_ylim([yf,yc])
    axis[iax].set_title(title)
    
def drawArrow00(axis,origin,vector,color='k',iax=0,label='None'):
    aw, lw = 0.08, 1.2                               # arrow width, line-width
    if label == 'None':
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw, 
            color=color, fill=False, length_includes_head=True)
    else:
        axis[iax].arrow(origin[0],origin[1],vector[0],vector[1],width=aw,lw=lw,
            color=color, label=label, fill=False, length_includes_head=True)

def sequence_trunkI(rows, clms,start = 0):
    '''Use interges to create a sequence trunk matrix'''
    ST = [j for j in range(start,clms+start)]
    for i in range(1,rows):
        si = start + i
        STr= [j for j in range(si,clms+si)] 
        ST = np.vstack((ST,STr))
    return ST

def sequence_trunkA(A, clms, start = 0):
    '''A: a given 1D np.array; to create sequence trunk matrix'''
    rows = int(A.shape[0]/clms)
    ST = [j for j in A[start:clms+start]]
    for i in range(1,rows):
        si = start + i
        STr= [j for j in A[si:clms+si]]
        ST = np.vstack((ST,STr))
    return ST

def null_exam(A): #, A_name='A'):
    '''Examination of numpy array A by computing its row-echolen-from. 
    input: array A; return: N, null-space of A; nd: nullity of A.
    '''
    #A_name=varName(A, globals())[0]
    A_name=varName2(A)[0]
    n = np.shape(A)
    M_rref = sp.Matrix(A).rref()
    # Basis for null-space (see the textbook by Gilbert Strong)
    r = np.size(M_rref[1])
    print(f"Row echelon form of {A_name} with pivot columns:{M_rref}; "\
          f'{A_name}.shape:',n,'\nRank=',r,f'or dimension of C({A_name})')
    nd = n[1] - r
    print(f"Dimension of the null-space N({A_name}), or nullity = {nd}")
    
    i_old = [i for i in range(n[1])]    # original variables order
    iFC = [i for i in i_old if i not in set(M_rref[1])]
    i_new = list(M_rref[1]) + list(iFC) # changed variables order
    print(f"Free columns = {iFC}, i_old= {i_old}, i_new= {i_new}")
    
    # Form the basis vectors for the Null space: N=[-F  I].T
    if nd == 0: 
        N = 0; print(f"Thus, the null-space has only the zero vector")
    else: 
        N =np.array(-M_rref[0][:r, iFC])   # free columns for pivot vars 
        N = np.append(N, np.eye(nd), axis=0)  # add I for free variables
        print(' N shape:',np.shape(N), 'I(nd).shape:', np.eye(nd).shape)
        print(f"Basis vectors for the null-space:\n{N}")
        
        # Swap rows, if variable order is changed 
        for i in range(n[1]):
            if i_new[i] != i_old[i]:
                N[[i_new[i], i_old[i]], :] = N[[i_old[i], i_new[i]], :]
                i_new[i_new[i]], i_new[i_old[i]] = \
                i_new[i_old[i]], i_new[i_new[i]]
                print(f"Basis vectors for null-space after swap:\n{N}")
                
        print(f"Transformed basis vector of the null-space by {A_name}:\
              \n{A.dot(N)}, N is orthogonal to all row-vecors of {A_name}.")
        print(f"N is also orthogonal to rref of {A_name}:\n{M_rref[0]@N}")
        
    return N, nd

def simplifyM(M):     
    '''To simplify a Sympy matrix, by factor() each of its elements'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i,j] = sp.factor(M[i,j]).subs(1.,1) # simplify the expression
            M[i,j] = sp.factor(M[i,j]).subs(1.,1) # simplify it again
    return M


def simplifyM0(M):     
    '''To simplify a Sympy matrix, by factor() each of its elements'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i,j] = M[i,j].simplify()  # simplify the expression
    return M

def simplifyMd(M):                      
    '''To deep-simplify a Sympy matrix with fractions, by working on
    each of its elements'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Mn,Md=sp.fraction(M[i,j])     # Simplify the expression
            Mn,Md=Mn.expand(),Md.expand() # for both numerator 
            Mn,Md=Mn.evalf(),Md.evalf()   # and denominator.
            M[i,j] = Mn/Md                # Put it back.
    return M

def simplifyMm(M):                        # added to grcords
    '''To multi-simplify a Sympy matrix with fractions, by working on
    each of its elements using factor(), expand(), evalf(), etc.'''
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):    # simplify the expression
            M[i,j] = sp.factor(M[i,j]).subs(1.0, 1) 
            M[i,j] = M[i,j].evalf().subs(1.0, 1)
            M[i,j] = sp.factor(M[i,j]).subs(1.0, 1)
            M[i,j] = sp.expand(M[i,j]).subs(1.0, 1)
            M[i,j] = sp.factor(M[i,j]).subs(1.0, 1)
    return M

def varName(var, allVarNames):
    '''return the name of a variable. Usage:varName(var,globals())[0]
    ref: https://stackoverflow.com/questions/18425225/
    getting-the-name-of-a-variable-as-a-string'''
    return [name for name in allVarNames if allVarNames[name] is var]

def varName2(var):
    '''return the name of a variable. Usage:varName2(var)[0]
    ref: https://stackoverflow.com/questions/18425225/
    getting-the-name-of-a-variable-as-a-string'''
    current_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [name for name, value in current_vars if value is var]

def cholesky(A):
    '''Cholesky decomposition.
    This is only a minimalistic code for principle demonstration. 
    Input:   A(n, n), SPD, numpy array 
    Returns: L(n, n), L lower triangular 
    Usage:   L = Cholesky(A) ''' 
    n = A.shape[0]                 # get the dimension of A
    L = A.copy().astype(A.dtype)
    for i in range(n):
        for j in range(i+1,n):
            L[j:,j] = L[j:,j]-L[j:,i]*L[j,i].conj()/L[i,i]
            #L[j,j:]=L[j,j:]-L[i,j:]*L[i,j].conj()/L[i,i] # upper
            
        L[i:,i]=L[i:,i]/np.sqrt(L[i,i])
        #L[i,i:]=L[i,i:]/np.sqrt(L[i,i]) # upper
    return np.tril(L, k = 0)

def gs_LU(A):
    '''LU decomposition via Gaussin Elimination without Pivoting.
    This is only a minimalistic code for principle demonstration. 
    Input:   A(n, n), numpy array 
    Returns: L(n, n), lower triangular; U(n, n), upper
    Usage:   L, U = gs_LU(A) ''' 
    n = A.shape[0]                 # get the dimension of A
    U = A.copy().astype(A.dtype)
    L = np.eye(n,n).astype(A.dtype)
    for i in range(n-1):
        for j in range(i+1,n):
            L[j,i] = U[j,i]/U[i,i]
            U[j,i:]=U[j,i:]-L[j,i]*U[i,i:]
    return L, U

def det_minor(A, i, j):                        # in grcodes.py
    '''compute the determinant of the minor_ij of an numpy array''' 
    return lg.det(np.delete(np.delete(A,i,axis=0), j, axis=1))

def pure_QR0(A, atol=1.e-8, n_iter=1000):
    '''The pure QR for the Schur factorization. To be further tested!!!. 
    A(n, n): Inputs, an numpy array 
    A(n, n): Schur form; and Q(n, n): Unitary matrix''' 
    m, n = A.shape                           # get the dimensions 
    dtypeA = A.dtype                         # get the data type of A
    Acopy = A.copy().astype(dtypeA)    
    Q = np.zeros((m, n), dtype=dtypeA)       # initialization
    R = np.zeros((n, n), dtype=dtypeA)
    Qr = np.eye(n, dtype=dtypeA)          
    
    for i in range(n_iter):                  # control by iterations 
        Q, R = mgs_QR(Acopy) # or lg.qr() & h_QR() for QR factorization
        Acopy = R@Q                 # Recombine factors in revise order
        Qr = Qr@Q                   # buid up Q
        if lg.norm(Acopy[n-1,:n-1]) < atol:  # control by accuracy
            print(f"Converged at iteration = {i}"); break   
    return Acopy, Qr

def pure_QR(A, atol=1.e-12, n_iter=1000):
    '''The pure QR for the Schur factorization . 
    A(n, n): Inputs, an numpy array 
    A(n, n): Schur form; and Q(n, n): Unitary matrix''' 
    m, n = A.shape; dtypeA = A.dtype   # get dimensions & data-type of A
    Qr = np.eye(n, dtype=dtypeA)          
    for i in range(n_iter):                  # control by iterations 
        Q, R = lg.qr(A)                      # QR factorization
        A = R@Q                     # Re-combine factors in revise order
        Qr = Qr@Q                            # buid up Q
        if lg.norm(A[n-1,:n-1]) < atol:      # control by accuracy
            print(f"Converged at iteration = {i}"); break   
    return A, Qr

def forward_subst(L, b):
    '''Primary code for solve Ly=b via forward substitution.
    Inputs:  L(n, n)-- lower triagonal matrix, numpy array 
             b(n)--vector on the right-handside
    Leturns: x(n)--solution
    usage:   x = forward_subst(L, b)''' 
    n, x = b.shape[0], np.zeros_like(b) 
    x[0] = b[0]/L[0,0]
    for i in range(1,n): 
        r = 0.
        for j in range(0,i):
            r += L[i,j]*x[j]
        x[i] = (b[i]-r)/L[i,i]
    return x

def back_subst(R, b):                        # in grcodes.py
    '''Primary code for solve Rx=b via backward substitution.
    Inputs:  R(n, n)-- upper triagonal matrix, numpy array 
             b(n)--vector on the right-handside
    Returns: x(n)--solution
    usage:   x = back_subst(R, b)''' 
    n, x = b.shape[0], np.zeros_like(b) 
    x[-1] = b[-1]/R[-1, -1]
    for i in range(n-2,-1,-1):
        r = 0
        for j in range(i+1,n):
            r += R[i,j]*x[j]
        x[i] = (b[i]-r)/R[i,i]
    return x

def find_orth(A):
    '''This finds a vector that orthogonal to a given matrix A of m byn.
    returns: the normalized vector. 
    https://stackoverflow.com/questions/50660389/generate-a-vector-\
    that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension'''
    x0 = np.random.randn(A.shape[0], 1)
    A = np.hstack((A, x0))
    b = np.zeros(A.shape[1])
    b[-1] = 1
    print(A.shape, A.shape[1], b.shape)
    x = lg.lstsq(A.T, b, rcond=None)[0]
    return x/lg.norm(x)

def mapAA(A, f_name='images/paralgramA.png',fig_size=4,font_size=5):
    '''Plot the mapped parallelograms by given matrix A[i,j].
    It maps a square to k quadrants.'''
    
    plt.rcParams.update({'font.size': font_size})
    I = np.array([[1, 0], 
                  [0, 1]])
    fig, axis = plt.subplots(1,1,figsize=(fig_size,fig_size)) 
    axis.set_aspect(1)
    #axis.set_title('Mapped the unit square') 
    axis.grid(color='r', linestyle=':', linewidth=0.5)

    x1 = [0, A[0,0], A[0,0]+A[0,1],A[0,1], 0]  
    x2 = [0, A[1,0], A[1,0]+A[1,1],A[1,1], 0]
    axis.plot(x1, x2, label="Maped by A")
    
    x1 = [0, I[0,0], I[0,0]+I[0,1],I[0,1], 0]  
    x2 = [0, I[1,0], I[1,0]+I[1,1],I[1,1], 0]
    axis.plot(x1, x2, label="Square")
    
    mpl.rcParams['text.usetex'] = True # False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    xloc = .01*fig_size; yloc = 0.2*fig_size
    dx = 3.8*font_size/72; dy = .6*font_size/72
    
    axis.text(xloc+dx, yloc+dy, str(A[0,0]))
    axis.text(xloc+dx+3.5*dy, yloc+dy, str(A[0,1]))
    
    axis.text(xloc+dx, yloc-1.2*dy, str(A[1,0]))
    axis.text(xloc+dx+3.5*dy, yloc-1.2*dy, str(A[1,1]))
    
    axis.text(xloc, yloc,\
              r'$\mathbf{A}=\begin{bmatrix}\;\;&\;\;\\&\end{bmatrix}$')
    
    axis.legend()#loc='lower center')#, bbox_to_anchor=(1.1, 0.9)) 
    plt.savefig(f_name, dpi=500)
    plt.show()

def lagrange_interp(x, y, x_eval):
    """Computes the Lagrange interpolating polynomial for a given set of 
    data points. Suggested by ChapGPT.
    ----------    
    Parameters
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points.
    x_eval : float or array-like
        The point(s) at which to evaluate the interpolating polynomial.
    -------        
    Returns
    p : float or array-like
        The value(s) of the Lagrange interpolating polynomial at x_eval.
    """
    n = len(x)
    p = 0.0
    
    for i in range(n):
        p_i = 1.0
        for j in range(n):
            if i == j:
                continue
            p_i *= (x_eval - x[j]) / (x[i] - x[j])
        p += y[i] * p_i
    
    return p


def lagrange_interp_diff(x, y, x_eval):
    """Computes the derivatives of the Lagrange interpolating polynomial 
    for a given set of data points. Suggested by ChapGPT. 
    ----------
    Parameters: 
    x : array-like
        The x-coordinates of the data points.
    y : array-like
        The y-coordinates of the data points.
    x_eval : float or array-like
        The point(s) at which to evaluate the derivatives of the 
        interpolating polynomial.
    -------
    Returns:
    p_deriv : float or array-like
        The derivative(s) of the interpolating polynomial at x_eval.
    """
    n = len(x)
    p_deriv = 0.0
    
    for i in range(n):
        p_i = 1.0
        p_i_deriv = 0.0
        
        for j in range(n):
            if i == j:
                continue
            p_i *= (x_eval - x[j]) / (x[i] - x[j])
            p_i_deriv += 1.0 / (x_eval - x[j])
        
        p_deriv += y[i] * p_i_deriv * p_i
    
    return p_deriv

## Example: 
#x = np.array([-1, 0, 1])  # Define a set of data points
#y = np.array([1, 0, 1])

## Evaluate interpolating polynomial and its derivative at some points
#x_eval = np.linspace(-1.5, 1.5, num=100)
#p = np.array([lagrange_interp(x, y, xi) for xi in x_eval])
#p_deriv = np.array([lagrange_interp_diff(x, y, xi) for xi in x_eval])

#plt.plot(x_eval, p, label='Lagrange Interpolating Polynomial')
#plt.plot(x_eval, p_deriv, label='Derivative of Interpolating Polynomial')
#plt.plot(x, y, 'ro', label='Data Points')
#plt.legend(); plt.show()

def centroid_I(hb, wft, wfb, tft, tfb, tw):
    '''Compute the centroid of an I-beam made of rectangles. 
    The reference coordinate is on the bottom edge with the left-bottom 
    corner as the origin. The code can be used for any composite area 
    with upto 4 rectangles. For T beam, for example, just set tfb=0. 
    The whole area must be left-right symmetric.
    Inputs: 
    hb:  beam height (total)
    wft: width of flange on top
    wfb: width of flange at bottom
    tft: thickness of flange on top
    tfb: thickness of flange at bottom
    tw:  web thickness
    Return: 
    x_bar, y_bar: the centroid  
    A: The total area
    Ai: areas of each rectangles
    '''
    width_ai  = np.array([wft, tw,         wfb])            
    height_bi = np.array([tft, hb-tft-tfb, tfb])            
    xi_bar    = np.array([wfb/2,  wfb/2,  wfb/2])
    yi_bar    = np.array([hb-tft/2, (hb-tft-tfb)/2+tfb, tfb/2])

    Ai = width_ai*height_bi
    A = np.sum(Ai)
    print(f"Areas of sub-areas = {Ai} (L^2)")
    print(f"Areas of I-beam = {A} (L^2)")

    x_bar = np.sum(xi_bar*Ai)/A
    y_bar = np.sum(yi_bar*Ai)/A

    print(f"The centroid (L)):\n x = {x_bar} \n y = {y_bar}")
    return x_bar, y_bar, A, Ai

def secondMoments_I(hb, wft, wfb, tft, tfb, tw, cxi, cyi, sy=True):
    '''Compute the second moments of beams made of rectangles. 
    The reference coordinate is on the bottom edge with the left-bottom 
    corner as the origin and x-axis on the bottom edge. 
    The code can be used for any composite area 
    with upto 3 rectangles. For T beam, for example, just set tfb=0. 
    Inputs: 
    hb:  beam height (total)
    wft: width of flange on top
    wfb: width of flange at bottom
    tft: thickness of flange on top
    tfb: thickness of flange at bottom
    tw:  web thickness
    cxi, cyi: centers of each sub-area wrt the common coordinates. 
    sy:  if True (I-, T-beam), cxi, cyi are not used, compute here in. 
         Otherwise, cxi, cyi will be used here in.
    Return: 
    Ixx, Iyy, Ixy: 2nd moments
    x_bar, y_bar: the centroid  
    A: The total area
    Ai: areas of each rectangles
    '''
    width_ai  = np.array([wft, tw,         wfb])   
    height_bi = np.array([tft, hb-tft-tfb, tfb])   
    
    # when whole area is symmetric with respect to y-axis:     
    if sy:    #compute the centers (cxi, cyi) for each sub-area. 
        cxi = np.array([wfb/2,  wfb/2,  wfb/2])
        cyi = np.array([hb-tft/2, (hb-tft-tfb)/2+tfb, tfb/2])

    print(f"Center of sub-areas (L):\n x = {cxi} \n y = {cyi}")
    
    Ai = width_ai*height_bi
    A = np.sum(Ai)
    print(f"Areas of sub-areas = {Ai} (L^2)")
    print(f"Areas of I-beam = {A} (L^2)")

    x_bar = np.sum(cxi*Ai)/A
    y_bar = np.sum(cyi*Ai)/A

    print(f"The centroid = ({x_bar}, {y_bar}) (L)")
    
    # shifts the coordinates to the centroid of the whole area:
    xi_bar = cxi - x_bar
    yi_bar = cyi - y_bar
    
    print(f"New center of sub-areas (L):\n x = {xi_bar}\n y = {yi_bar}")
    
    Ixx = np.sum(width_ai*height_bi**3/12 + yi_bar**2 * Ai)
    Iyy = np.sum(height_bi*width_ai**3/12 + xi_bar**2 * Ai)
    Ixy = np.sum(xi_bar*yi_bar * Ai)
    
    print(f"2nd moments (L)^4:\n Ixx = {Ixx}\n Iyy = {Iyy}\n Ixy = {Ixy}")
    
    return Ixx, Iyy, Ixy, x_bar, y_bar, xi_bar, yi_bar, A, Ai

def Ixx2IXX(Ixx, Iyy, Ixy, 𝜃):
    '''Coordinate transformation for second moments of area.
    Ixx, Iyy, Ixy: for origina (x,y) 
    IXX, IYY, IXY: for (X,Y) that is 𝜃-rotated from (x,y)
    '''
    c = np.cos(𝜃); s = np.sin(𝜃)
    c2= c**2;      s2= s**2;  cs = c*s
    IXX = Ixx*c2 + Iyy*s2 - 2*Ixy*cs 
    IYY = Ixx*s2 + Iyy*c2 + 2*Ixy*cs 
    IXY = (Ixx-Iyy)*cs + Ixy*(c2 - s2) 
    
    return IXX, IYY, IXY

def q_web(y, Q_Dweb, tw, hwu, hwl):
    '''computing the shear flow q at any location s in the web of 
    an I-beam.
    Q_Dweb: Q value of the web at the top junction
    tw: thickness of the web
    hwu: height of the web above the neutral axis
    hwl: height of the web blow  the neutral axis
    Example: Computing the force in the web of I-beam. 
    # Because q_web() has parameters, use partial() to fix the parameters: 
    q_web_params = partial(q_web, Q_Dweb=Q_Dweb, tw=tw, hwu=hwu, hwl=hwl)
    F_webG, error = si.quad(q_web_params, hwl, hwu)       # use Gauss int.
    print(f"Force in the web (kN): {F_webG*1e-3}, intgral error= {error}")
    print(f"The true shear sorce V (kN): {V*1e-3}")
    One may use a lambda function. for this, if it is used just for once.
    '''
    Q_sweb = Q_Dweb + tw*(hwu - y)*(hwu + y)/2
    q_s_web = V/Izz * Q_sweb
    return q_s_web
 

def solver1D4(E, I, by, l, v0, 𝜃0, vl, 𝜃l, V0, M0, Vl, Ml, key='c-c'):
    '''Solves the Beam Equation for integrable distributed body force: 
           u,x4=-by(x)/EI, with various boundary conditions (BCs): 
           c-c, s-c, c-s, f-c, c-f. 
    Input: EI: bending stiffness factor; by, body force; l, the length 
           of the beam; v0, 𝜃0, V0, M0, deflection, rotation, 
           shear force, moment at x=0; vl, 𝜃l, Vl, Ml, those at x=l. 
    Return: u, v_x, v_x2, v_x3, v_x4   upto 4th derivatives of v''' 
    c0, c1, c2, c3 = sp.symbols('c0, c1, c2, c3')   #integration constant
    x = sp.symbols('x') # x coordinate
    EI = E*I            # I is the Iz in our choosen coordinates. 
    # Integrate 4 times:     
    v_x3= sp.integrate(by/EI,(x, 0, x))+ c0  #ci: integration constant   
    v_x2= sp.integrate( v_x3,(x, 0, x))+ c1          
    v_x = sp.integrate( v_x2,(x, 0, x))+ c2   
    v   = sp.integrate(  v_x,(x, 0, x))+ c3   
    
    # Solve for the 4 integration constants: 
    if   key == "s-s":
        cs=sp.solve([v.subs(x,0)-v0, v_x2.subs(x,0)-M0/EI,
                     v.subs(x,l)-vl, v_x2.subs(x,l)-Ml/EI],[c0,c1,c2,c3])
    elif key == "c-c":
        cs=sp.solve([v.subs(x,0)-v0, v_x.subs(x,0)-𝜃0,
                     v.subs(x,l)-vl, v_x.subs(x,l)-𝜃l], [c0,c1,c2,c3])
    elif key == "s-c":
        cs=sp.solve([v.subs(x,0)-v0, v_x2.subs(x,0)-M0/EI,
                     v.subs(x,l)-vl, v_x.subs(x,l)-𝜃l], [c0,c1,c2,c3])
        #print('solver1D4:',cs[c0],cs[c1],cs[c2],cs[c3])
    elif key == "c-s":
        cs=sp.solve([v.subs(x,0)-v0, v_x.subs(x,0)-𝜃0,
                     v.subs(x,l)-vl, v_x2.subs(x,l)-Ml/EI],[c0,c1,c2,c3])
    elif key == "c-f":
        cs=sp.solve([v.subs(x,0)-v0, v_x.subs(x,0)-𝜃0,
           v_x3.subs(x,l)+Vl/EI, v_x2.subs(x,l)-Ml/EI], [c0,c1,c2,c3])
    elif key == "f-c":
        cs=sp.solve([v_x3.subs(x,0)-V0/EI, v_x2.subs(x,0)-M0/EI,
                     v.subs(x,l)-vl, v_x.subs(x,l)-𝜃l], [c0,c1,c2,c3])
    else:
        print("Please specify boundary condition type.")
        sys.exit()
        
    # Substitute the constants back to the integral solutions   
    v   =   v.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v   =   v.expand().simplify().expand()
    
    v_x = v_x.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v_x = v_x.expand().simplify().expand()
    
    v_x2=v_x2.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v_x2=v_x2.expand().simplify().expand()
    
    v_x3=v_x3.subs({c0:cs[c0],c1:cs[c1],c2:cs[c2],c3:cs[c3]})
    v_x3=v_x3.expand().simplify().expand()
    
    v_x4 = sp.diff(v_x3,x).expand()
    
    print("Outputs form solver1D4(): v, 𝜃, Mx, Vx, qy")
    return v,v_x,(EI*v_x2).expand(),(-EI*v_x3).expand(),(v_x4).expand()

def create_DPF3D(order):
    '''Creates polynomial Laplace basis functions that satisfies 
       ∇^2 = 0 with a desirved order that must be >=2.
       '''
    x, y, z = sp.symbols("x, y, z")
    𝜓 = sp.Function("𝜓")(x, y, z)           # a displacement-function
    c = sp.symbols("c0:99")    # coeffs, define more than what we need
    
    # create tri-nomials of "order"
    basis = Matrix([x**i*y**j*z**k for i in range(order+1) for j in 
            range(order+1) for k in range(order+1) if i+j+k==order])
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")
    
    cs = Matrix([c[i] for i in range(len(basis))])     # coeffs needed
    num_cs = (order+2)*(order+1)/2
    print(f'order={order}, No. of constants={len(cs)}, ' 
          f'should be (order+2)*(order+1)/2: {num_cs}')
    
    𝜓 = (cs.T@basis)[0]    # create the function by linear combination
    #print(f"Initial 𝜓 with complete order {order} for 𝜓:\n {𝜓}")
    
    # form the order-reduced polynomial bases after applying ∇^2:
    basis2 = Matrix([x**i*y**j*z**k for i in range(order+1) for j in 
            range(order+1) for k in range(order+1) if i+j+k==order-2])
    #print(f"Polynomials bases after ∇^2:\n {basis2.T} {len(basis2)}")
    
    nab2𝜓=(𝜓.diff(x,2) + 𝜓.diff(y,2) + 𝜓.diff(z,2)).expand() # ∇^2

    # collect terms for order-reduced bases, to group the coeffs  
    for i in range(len(basis2)):       
        nab2𝜓 =  nab2𝜓.collect(basis2[i]) #print(nab2𝜓)
    
    # form equations of relations of the coeffs. 
    eqs = [nab2𝜓.coeff(basis2[i]) for i in range(len(basis2))]  
    if len(eqs)==1: eqs = [nab2𝜓]
        
    sln_cs=sp.solve(eqs, cs)  #print(sln_cs)
    keys_sln = list(sln_cs.keys())                     # solution keys
    
    # get the remaining coefficients in the expression
    keys_csi  = list(filter(lambda x: x not in keys_sln, cs))
    ns = order*(order-1)/2
    print(keys_sln,"\n", len(keys_sln),"is order*(order-1)/2:",ns,"\n",
          keys_csi,"\n", len(keys_csi),"should be 2*order+1")
    
    𝜓_ci = 𝜓.subs(sln_cs).expand()           # get the 𝜓 with keys_csi
    
    # collect terms based on keys_csi, to group polynomials  
    for i in range(len(keys_csi)):
        𝜓_ci = 𝜓_ci.collect(keys_csi[i]) 
    
    # group polynomials, based on keys_csi
    DPFs=Matrix([(𝜓_ci.coeff(keys_csi[i])).expand() 
                 for i in range(len(keys_csi))]) 
    
    nab2𝜓=(DPFs.diff(x,2)+DPFs.diff(y,2)+DPFs.diff(z,2)).expand() #∇^2
    
    print(f'Is ∇^2 𝜓=0? {nab2𝜓}!\n  The number of bases = {len(DPFs)}')
    return DPFs

def create_BBFs(order):
    '''Creates polynomial biharmonic basis functions that satisfies 
       ∇^4=0 with a desirved order that must be larger or equal to 4.
       '''
    x, y = sp.symbols("x, y")
    𝜙 = sp.Function("𝜙")(x, y)               # a displacement-function
    c = sp.symbols("c0:20")    # coeffs, define more than what we need
    cs = Matrix([c[i] for i in range(order+1)]) #  coefficients needed
    print(f"order:{order}, terms:{len(cs)}, should be (order+1)")

    # form the complete polynomial bases for 𝜙: 
    basis = Matrix([x**i * y**(order-i) for i in range(order+1)])
    #print(f"Polynomials bases, complete order {order}:\n {basis.T}")

    # form the remaining polynomial bases after applying ∇^4:
    basis4 = Matrix([x**i * y**(order-4-i) for i in range(order-4+1)])
    #print(f"Polynomials bases after applying ∇^4:\n {basis4.T}")
    
    𝜙 = (cs.T@basis)[0]    # create the function by linear combination
    #print(f"Initial 𝜙 with complete order {order} for 𝜙:\n {𝜙}")

    nab2𝜙=(𝜙.diff(x,2) + 𝜙.diff(y,2)).simplify()                # ∇^2
    nab4𝜙=(nab2𝜙.diff(x,2)+nab2𝜙.diff(y,2)).simplify()          # ∇^4 

    # collect terms for remaining bases, so that the coeffs are grouped. 
    for i in range(len(basis4)):       
        nab4𝜙 =  nab4𝜙.collect(basis4[i]) 

    # form (order+1-4) equations of relations of coeffs. 
    eqs = [nab4𝜙.coeff(basis4[i]).expand() for i in range(len(basis4))]
    if len(eqs)==1: eqs = [nab4𝜙]
    sln_cs=sp.solve(eqs, cs)   
    keys_sln = list(sln_cs.keys())                     # solution keys
    
    # get the remaining coefficients in the expression
    keys_csi  = list(filter(lambda x: x not in keys_sln, cs))
    print(keys_sln, len(keys_sln),",  should be (order-4+1)","\n",
          keys_csi, len(keys_csi),",  should be:", 4)

    𝜙_final = 𝜙.subs(sln_cs).expand()               # get the final 𝜙

    # collect terms based on remaining coeffs, polynomials are grouped. 
    for i in range(len(keys_csi)):
        𝜙_final = 𝜙_final.collect(keys_csi) 

    # bases that satisfies ∇^4 = 0. They are with the last 4 coeffs:
    BBFs=Matrix([(𝜙_final.coeff(keys_csi[i])).expand() 
                 for i in range(len(keys_csi))]) 

    # check the results: 
    nab2𝜙=(BBFs.diff(x,2)+BBFs.diff(y,2)).expand()               # ∇^2
    nab4𝜙=(nab2𝜙.diff(x,2)+nab2𝜙.diff(y,2)).expand()            # ∇^4 
    print(f'Is ∇^4𝜙 = 0? {nab4𝜙.T}! Number of bases: {len(nab4𝜙)}')
    return BBFs

def field_plot(expr, x_mesh, z_mesh, title=''):
    """Plot the field, by Guarín-Zapata, Nicolás, under MIT license.
    at https://github.com/nicoguaro/continuum_mechanics
    """

    # Lambdify the function
    var = [x, z]
    arr = [x_mesh, z_mesh]
    vals = OrderedDict(zip(var, arr)).values()
    expr_fun = lambdify(var, expr, "numpy")
    expr_mesh = expr_fun(*vals)

    # Determine extrema
    vmin = np.min(expr_mesh)
    vmax = np.max(expr_mesh)
    print("Minimum value in the domain: {:g}".format(vmin))
    print("Maximum value in the domain: {:g}".format(vmax))
    vmax = max(np.abs(vmax), np.abs(vmin)) 
    #print(f"log10(vmax)={vmax},{float(vmax)}")
    vmax = float(vmax)

    # Plotting
    fig = plt.gcf()
    levels = np.logspace(-1, np.log10(vmax), 20)
    levels = np.hstack((-levels[-1::-1], [0], levels))
    cbar_ticks = ["{:.2g}".format(level) for level in levels]
    cont = plt.contourf(x_mesh, z_mesh, expr_mesh, levels=levels,
           cmap="RdYlBu_r", norm=colors.SymLogNorm(0.1)) #,base=10))
    cbar = fig.colorbar(cont, ticks=levels[::2])
    cbar.ax.set_yticklabels(cbar_ticks[::2])
    plt.axis("image")
    plt.gca().invert_yaxis()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.title(title)
    return cont

def contour_plot(X, Y, Z, x_lbl='x-axis', y_lbl='y-axis',title='title',lvls=40):
    
    '''Generate a simple contour plot for given X, Y, Z
    '''
    vmin = np.nanmin(Z)                                 # find extrema
    vmax = np.nanmax(Z)
    print(f"Minimum value in the domain: {vmin:g}")
    print(f"Maximum value in the domain: {vmax:g}")
    
    fig, ax = plt.subplots()
    cntur = ax.contourf(X, Y, Z, cmap='RdYlBu_r',levels=lvls) #'viridis'
    plt.colorbar(cntur, label='Color Value')

    ax.set_aspect('equal')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title(title)
    
    return cntur

def curvaturef(F):
    '''Compute the curvature an implicit expression of a curve: F(x,y)=0.
    return: curvature
    Example: 
    from sympy import symbols, diff 
    x, y, a, b = symbols('x, y, a, b')                # Define symbols
    F = x**2/a**2 + y**2/b**2 - 1            # Define F for an ellipse
    curvature = curvaturef(F); print(curvature) # should be: 
    #Abs(8*y**2/(a**2*b**4)+8*x**2/(a**4*b**2))/(4*y**2/b**4+4*x**2/a**4)**1.5
    '''
    f_x = diff(F, x)                               # first derivatives
    f_y = diff(F, y)
    
    f_xx = diff(f_x, x)                           # second derivatives
    f_yy = diff(f_y, y)
    f_xy = diff(f_x, y)
    
    curvature = abs(f_xx*f_y**2+f_x**2*f_yy)/(f_x**2+f_y**2)**(3/2) 
    return curvature            # ⇑ the curvature formula


def hyper2e(x, s='s'):
    '''produces exponential form of hyperbolic functions for given x'''
    if s == 's': return (sp.exp(x) - sp.exp(-x))/2
    if s == 'c': return (sp.exp(x) + sp.exp(-x))/2
    if s == 't': return (sp.exp(x) - sp.exp(-x))/(sp.exp(x)+sp.exp(-x))
    if s =='ct': return (sp.exp(x) + sp.exp(-x))/(sp.exp(x)-sp.exp(-x))
    
def hyper2to1_dic(x):
    '''produces a dictionary for trigonometric expansion of 
    4 hyperbolic functions
    '''
    dic = {sp.sinh(x): sp.expand_trig(sp.sinh(x)),
           sp.cosh(x): sp.expand_trig(sp.cosh(x)),
           sp.tanh(x): sp.expand_trig(sp.tanh(x)),
           sp.coth(x): sp.expand_trig(sp.coth(x))}
    return dic

def hyper2e_dic(𝜉0, 𝜉, 𝜂):
    '''produces a dictionary for converting hyperbolic functions to
       exponential functions
    '''
    dic_e={sp.cosh(2*𝜉-2*𝜉0+2*sp.I*𝜂): hyper2e(2*𝜉-2*𝜉0+2*sp.I*𝜂, 'c'),
           sp.sinh(2*𝜉-2*𝜉0+2*sp.I*𝜂): hyper2e(2*𝜉-2*𝜉0+2*sp.I*𝜂, 's'),
           sp.cosh(2*sp.I*𝜂-2*𝜉-2*𝜉0): hyper2e(2*sp.I*𝜂-2*𝜉-2*𝜉0, 'c'),
           sp.sinh(2*sp.I*𝜂-2*𝜉-2*𝜉0): hyper2e(2*sp.I*𝜂-2*𝜉-2*𝜉0, 's'),
           sp.cosh(2*sp.I*𝜂+2*𝜉): hyper2e(2*sp.I*𝜂+2*𝜉, 'c'),
           sp.sinh(2*sp.I*𝜂+2*𝜉): hyper2e(2*sp.I*𝜂+2*𝜉, 's'),
           sp.cosh(𝜉-sp.I*𝜂): hyper2e(𝜉-sp.I*𝜂, 'c'),
           sp.sinh(𝜉-sp.I*𝜂): hyper2e(𝜉-sp.I*𝜂, 's'),
           sp.cosh(𝜉+sp.I*𝜂): hyper2e(𝜉+sp.I*𝜂, 'c'),
           sp.sinh(𝜉+sp.I*𝜂): hyper2e(𝜉+sp.I*𝜂, 's'),
           sp.tanh(𝜉+sp.I*𝜂): hyper2e(𝜉+sp.I*𝜂, 't'),
         
           sp.cosh(𝜉0-sp.I*𝜂): hyper2e(𝜉0-sp.I*𝜂, 'c'),
           sp.sinh(𝜉0-sp.I*𝜂): hyper2e(𝜉0-sp.I*𝜂, 's'),
           sp.cosh(𝜉0+sp.I*𝜂): hyper2e(𝜉0+sp.I*𝜂, 'c'),
           sp.sinh(𝜉0+sp.I*𝜂): hyper2e(𝜉0+sp.I*𝜂, 's'),
           sp.tanh(𝜉0+sp.I*𝜂): hyper2e(𝜉0+sp.I*𝜂, 't')}
    return dic_e

def hyper02e_dic(𝜉0):
    '''produces a dictionary for substituting hyperbolic functions with
       exponential functions
    '''
    dic={sp.cosh(𝜉0):hyper2e(𝜉0,'c'), sp.cosh(4*𝜉0):hyper2e(4*𝜉0,'c'),
         sp.sinh(𝜉0):hyper2e(𝜉0,'s'), sp.sinh(4*𝜉0): hyper2e(4*𝜉0,'s'),
         sp.cosh(2*𝜉0): hyper2e(2*𝜉0, 'c'),
         sp.sinh(2*𝜉0): hyper2e(2*𝜉0, 's'),
         sp.tanh(𝜉0): hyper2e(𝜉0, 't'), sp.coth(𝜉0): hyper2e(𝜉0, 'ct')}
    return dic

#def eI_sc(alpha):
#    return {sp.exp(sp.I*alpha): sp.cos(alpha)+sp.I*sp.sin(alpha)}

def deepsimp(expr):
    '''perform deep simplification to a complex expression
    ''' 
    # Extract real and imaginary parts
    re_part = sp.re(expr); im_part = sp.im(expr)

    # Simplify each part
    simp_re_part = sp.simplify(re_part, deep=True).expand().simplify()
    simp_im_part = sp.simplify(im_part, deep=True).expand().simplify()

    # Reconstruct the simplified complex expression
    return simp_re_part + I * simp_im_part

def print_limit(expr_name, expression, max_line_length):
    '''print a variable/expression. Each line has a max_line_length.
    input: expr_name, the name of the expression.
    Example: 
    expr = 𝜎*(exp(4*𝜉) - 2*exp(2*𝜉)*cos(2*𝜂) + 1)*(-8*exp(2*𝜉0))  
    print_limit("expr = ", expr, 70) 
    '''
    print(expr_name)                   # print this a the beginning 
    for i in range(0, len(str(expression)), max_line_length):
        print(str(expression)[i:i+max_line_length])
        
def printLimit(expr_name, expression, max_line_length):
    '''print a variable/expression. Each line as a max_line_length, 
       and breaks only at a word.
    Example: 
    expr = 𝜎*(exp(4*𝜉) - 2*exp(2*𝜉)*cos(2*𝜂) + 1)*(-8*exp(2*𝜉0))  
    printLimit("expr = ", expr, 70) 
    '''
    print(expr_name, end='')
    words = str(expression).split()
    current_line_length = 0
    for word in words:
        if current_line_length + len(word) <= max_line_length:
            print(word, end=" ")
            current_line_length += len(word)+1  # +1: space btwn words
        else:
            print()  # start a new line
            print(word, end=" ")
            current_line_length = len(word) + 1

def writeRead(folderName, fileName, expression): 
    '''Write a variable/expression into a txt file with the same 
       fillename and then read the same variable/expression back in.
    inputs: folderName, fileName: all string.
    return: variable/expression: Python/Sympy expression
    '''
    with open(folderName+"/"+fileName+".txt","w",encoding="utf-8") as file:
        file.write(str(expression))

    with open(folderName+"/"+fileName+".txt","r",encoding="utf-8") as file:
        read_expression = file.read()

    return eval(read_expression)  # convert back to variable/expression

def xy2𝜉𝜂(x, y, c):
    '''inverse mapping from z-plane to 𝜁-plane'''
    z = x + 1j*y                        # complex z-plane in Cartesian
    𝜁 = np.arccosh(z/c)               # to complex 𝜁-plane in elliptic
    return 𝜁.real, 𝜁.imag 

def readFrFile(folderName, fileName, oldStr, newStr): 
    '''Read a variable/expression in a txt file.
    inputs: folderName, fileName: all string.
            oldStr in the expression is to be replace by newStr
    return: variable/expression: Python/Sympy expression
    '''
    with open(folderName+"/"+fileName+".txt","r",encoding="utf-8") as file:
        expression_ = file.read()

    for i in range(len(oldStr)):
        expression_ = expression_.replace(oldStr[i], newStr[i])
    return expression_   # convert back to variable/expression

def writeToFile(folderName, fileName, expression): 
    '''Write a variable/expression into a txt file with the same 
       fillename. Inputs: folderName, fileName: all string.
    '''
    with open(folderName+"/"+fileName+".txt","w",encoding="utf-8") as file:
        file.write(str(expression))

def readFromFile(folderName, fileName): 
    '''Read a variable/expression in a txt file.
    inputs: folderName, fileName: all string.
    return: variable/expression: Python/Sympy expressionSS
    '''
    with open(folderName+"/"+fileName+".txt","r",encoding="utf-8") as file:
        read_expression = file.read()

    return read_expression    # convert back to variable/expression

def stressOnN(S,N):
    '''Compute the traction (stress vector), normal stress, and \
    magnitude of the shear stress on a surface with normal vector N
    Input S: stress tensor in matrix form with 9 components\
          N: normal vector of the surface
    Return: S_N (traction), S_NN (normal stress), S_NS (shear stress)
    '''

    N = N/np.linalg.norm(N) # normalize vector N -> unit normal vector

    # Compute the traction vector on the surface with normal N
    S_N = np.dot(N,S)
    
    # Compute the normal stress on surface in the normal N direction
    S_NN = np.dot(S_N,N)
    
    # compute the magnitude of the shear stress on the surface
    S_NS = np.sqrt(S_N.dot(S_N)-S_NN**2)
    
    return S_N, S_NN, S_NS

def eigv_signAlter(St):
    '''Compute the eigenvalue and eigenvectors of stress tensor, St
       And the alter the signs of eigenvectors to see the effects. 
    Input: St, stress tensor, array like, 3 by 3.
    return: S_recovered stress tensor. 
    # Examples:     
    #St= np.array([[-10, 15, 0], [15, 30, 0], [0, 0, 0]])   # stresses
    #St =  np.array([[10, 15, 0], [15, -30, 0], [0, 0, 0]]) 
    #St =  np.array([[10, 15, 8], [15, -30, 5], [8, 5, 9]])  
    #S_recovered = eigv_signAlter(St)
    #S_recovered = eigv_signAlter(St)
    '''
    eigenValues, eigenVectors = principalS(St)
    signs = [(1., 1), (-1., 1), (1., -1),(-1., -1)]
    eigenVectors_=np.empty_like(eigenVectors)
    for i in range(len(signs)): 
        eigenVectors_[:,0] = signs[i][0]*eigenVectors[:,0]
        eigenVectors_[:,1] = signs[i][1]*eigenVectors[:,1]
        eigenVectors_[:,2] = eigenVectors[:,2]
        #eigenVectors_[:,2] = np.cross(eigenVectors_[:,0], eigenVectors_[:,1])
        S_recovered = eigenVectors_.T@St@eigenVectors_
        ei_diff = np.array([S_recovered[i,i]-eigenValues[i] 
                            for i in range(len(eigenValues))])
        diff_sum = np.sum(np.abs(ei_diff))
        print(f'i={i}, {signs[i][1]} diff_sum={diff_sum}')
        #if diff_sum < 1.e-129: 
        #    print(f'i={i}, diff_sum={diff_sum}')
        #    break

    print(f'Principal stress directions:\n{eigenVectors_}\n')
    return S_recovered 
