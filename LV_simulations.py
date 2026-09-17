import numpy as np
import jax
import jax.numpy as jnp
import scipy.integrate as scint
from scipy.stats import norm
import matplotlib.pyplot as plt
import diffrax as df
#General paramters choice for the simulations

#Choice of the case to simulate :0 for Elliptic , 1 for Wishart
choice =int(input("Choice of the case to simulate: 0 for Elliptic , 1 for Wishart: "))
if choice==0:#Elliptic case
    n=10             #size of the matrix
    mat_size=(n,n)
    #correlation paramter rho^2<=1: examples 1 GOE case, 0 i.i.d. case, -1 Skew-simmetric case
    rho=float(input("Choice of the correlation parameter,between -1 and 1: "))
    ex_par=rho
    kappa=2.001
    kappa_list=np.linspace(2.001,6,30)

################################################################################

    #Function for the computation of delta using the other parameters

    def delta_calc(kappa,gamma,rho):
        disc=(kappa)**2-4*rho*gamma
        if disc>=0:
            return (kappa+jnp.sqrt(disc))/2
        else:
            quit()
#############################################################################
    def dens_coeff(delta,gamma,rho):
        return 1+rho*gamma/(delta)**2


elif choice==1:#Wishart case
    n=1000              #size of the matrix
    p=300
    mat_size=(p,n)
    c=n/p
    ex_par=c
    kappa=8
    kappa_list=jnp.linspace(7.985,12,30)
    ####################################################


    #Function for the computation of delta using the other parameters

    def delta_calc(kappa,gamma,c):
        disc=(kappa-1-c*gamma)**2-4*c*gamma
        if disc>=0:
            return (kappa-1-c*gamma+jnp.sqrt(disc))/2
        else:
            quit()
    ####################################################
    def dens_coeff(delta,gamma,c):# the two extra input variables are only here to mantain the same structure  as the function dens_coeff in the first case
        return 1+1/delta
else:
    print("Error, the inserted value doesn't meet the requests!! Please restart the simulation and try again!")
    quit()
################################################################################
#Function for the solution of the system of paramters
def Sist_sol(kappa,gamma,sigma,ex_par,tol,choice):

    delta=delta_calc(kappa,gamma,ex_par)
    for i in range(10000):
        sigma_0=sigma
        gamma_0=gamma
        gamma= scint.quad(lambda r: norm.cdf(r/sigma_0), 0, 1)[0]

        # E[(sigma*Z+r)_+^2] = sigma^2 * E_r[(a^2+1)*Phi(a)+a*phi(a)], a=r/sigma where r is U(0,1)
        def integrand(r):
            a = r/sigma_0
            return (a**2+1)*norm.cdf(a) + a*norm.pdf(a)
        I = scint.quad(integrand, 0, 1)[0]
        if choice==0:
            c=1
        else:
            c=ex_par
        sigma = (1/delta) * np.sqrt(c*sigma_0**2*I)
        delta=delta_calc(kappa,gamma,ex_par)
        if(np.abs(sigma-sigma_0)<=tol and np.abs(gamma-gamma_0)<=tol):
            break
    return [delta, gamma, sigma]
################################################################################
#Function defining the Lotka-Volterra system
################################################################################

def LV_fun(t,y,args):# x is the variable of the function, G is the the interaction matrix of the Lotka-Volterra system
    r_n,Gamma_n=args
    return jnp.multiply(y,r_n-jnp.matmul(jnp.eye(n)-Gamma_n,y)+0*t)

#Generation of the random matrix of prescribed sizes mat_size depending on the chosen model
def matrix_gen(choice,ex_par,kappa,mat_size,key):
    X=jax.random.normal(key,mat_size)
    if choice==0:

        U=(X+X.T)/(jnp.sqrt(mat_size[0]))  # Symmetric part of the elliptic matrix
        V=(X-X.T)/(jnp.sqrt(mat_size[0]))  # Skew-symmetric part of the elliptic matrix
        Gamma_n=(jnp.sqrt(1+ex_par)*U+jnp.sqrt(1-ex_par)*V)/(2*kappa)#Elliptic random matrix

    else:

        Gamma_n=jnp.matmul(X.T,X)/(kappa*mat_size[0])#construction of the Wishart normalized matrix
    return Gamma_n

#Simulation of the Lokta-Volterra system with interaction matrix Gamma_n, and
@jax.jit
def LV_sim(r_n,Gamma_n,tol,key,y0):

    term=df.ODETerm(LV_fun)
    solver = df.Tsit5()
    saveat = df.SaveAt(t1=True)
    stepsize_controller = df.PIDController(rtol=1e-4, atol=1e-7)

    sol=df.diffeqsolve(
        term, solver, 0.0, 100.0, 0.01, y0,
        args=(r_n, Gamma_n), saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000_000,
    )

    final= sol.ys[-1]
    equ1=jnp.where(final>tol,final,0.0)
    surv_perc=jnp.count_nonzero(equ1)/n
    return [surv_perc,equ1]

#Evaluation of the theoretical density of the equilibrium
def theo_dens(y, delta,gamma, sigma, kappa,ex_par):
    coeff=dens_coeff(delta,gamma,ex_par)
    teo_dens = np.zeros_like(y)
    cond = y > 0
    z =  y[cond]/coeff
    teo_dens[cond] = (norm.cdf(z / sigma) - norm.cdf((z - 1.0) / sigma))/ (gamma*coeff)
    return teo_dens

################################################################################
#Here start the actual simulations
################################################################################
m=1 #number of Montecarlo simulations
tol=1e-6
gamma=0.5#initial value of gamma: proportion of surviving species
sigma=0.2#initial gamma: variance
key0 = jax.random.PRNGKey(0)
subkey = jax.random.split(key0,m)
r_n=jax.random.uniform(key0,(n,))

if choice == 0:
  kappa_fixed = 2.2
else:
  kappa_fixed = 8.5
y0=jnp.ones(n)#initial value for the Lotka-Volterra ODE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
#Surviving species plot
gamma_teo=[]
gamma_emp=[]

for kappa in kappa_list:

    sol=Sist_sol(kappa,gamma,sigma,ex_par,tol,choice)
    gamma_teo.append(sol[1])
    gamma_temp=[]

    for key in subkey:

        Gamma_n=matrix_gen(choice,ex_par,kappa,mat_size,key)
        gamma_temp.append(LV_sim(r_n,Gamma_n,tol,key,y0)[0])
    gamma_emp.append(np.mean(gamma_temp))

ax1.plot(kappa_list,gamma_teo)
ax1.scatter(kappa_list,gamma_emp,marker="h",color='darkgreen')

#Equilibrium distribution plot
equ_temp=[]
for key in subkey:
    Gamma_n=matrix_gen(choice,ex_par,kappa_fixed,mat_size,key)
    equ_temp.append(LV_sim(r_n,Gamma_n,tol,key,y0)[1])
equ_temp=np.array(equ_temp).flatten()
equ_emp=equ_temp[equ_temp>1e-6]
y=np.linspace(1e-5,equ_emp.max()*1.1,300)
delta=delta_calc(kappa_fixed,gamma,ex_par)
sist_sol=Sist_sol(kappa_fixed,gamma,sigma,ex_par,tol,choice)
equ_teo=theo_dens(y, sist_sol[0], sist_sol[1], sist_sol[2], kappa_fixed,ex_par)
ax2.hist(equ_emp,bins=40,density=True,color='darkgreen',edgecolor='black')
ax2.plot(y,equ_teo)
ax1.set_xlabel(r'Interaction strength ($\kappa$)')
ax1.set_ylabel(r'Proportion of surviving species ($\gamma$)')
ax2.set_xlabel(r'$x^*$')
ax2.set_ylabel('Density of equilibrium distribution')
ax2.set_title('Density of a surviving species at equilibrium')
if(choice==0):
    if(rho==1):
        ax1.set_title('Behaviour of the surviving species in the GOE case')
        plt.title('Empirical vs theoretical results for the Lotka-Volterra system in the GOE case')
        plt.tight_layout()
        plt.show()
        fig.savefig('LV_GOE.png')
    elif(rho==0):
        ax1.set_title('Behaviour of the surviving species in the I.I.D. case')
        plt.title('Empirical vs theoretical results for the Lotka-Volterra system in the i.i.d. case')
        plt.tight_layout()
        plt.show()
        fig.savefig('LV_iid.png')
    else:
        ax1.set_title('Behaviour of the surviving species in the Elliptic case with $ρ=%1.2f$'%ex_par)
        plt.title('Empirical vs theoretical results for the Lotka-Volterra system in the in the Elliptic case with $ρ=%1.2f$'%ex_par)
        plt.tight_layout()
        plt.show()
        fig.savefig('LV_ell.png')
else:
    ax1.set_title('Behaviour of the surviving species in the Wishart case')
    plt.title('Empirical vs theoretical results for the Lotka-Volterra system in the Wishart case')
    plt.tight_layout()
    plt.show()
    fig.savefig('LV_Wishart.png')
