clear all
clc
disp('To solve 1D convection diffusion equation by QUICK differencing scheme')
%% Input the values
F=1; %Value of convective flux
gamma=0;
nx=1001; %input('Enter the no. of division of x');
xl=1; %input('Enter the length of x');
dx=xl/(nx);
D=gamma/dx; %Value of diffusive flux
%% Input the Boundary conditions
phi_i=1; %input('Enter the initial value of phi');
phi_f=0.1; %input('Enter the final value of phi');
%% Define the variables
a=(4*D)+(7/8)*F;
b=-(4/3*D-3/8*F);
c=(2*D+3/8*F);
d=-(D+F);
e=-(D-3/8*F);
f=-(D+7/8*F);
g=(1/8)*F;
h=4*D-(3/8)*F;
m=-(4/3*D+6/8*F);
j=8/3*D+10/8*F;
k=-1/4*F;
l=8/3*D-F;
%% Matrix formation
A=full(gallery('tridiag',nx,f,c,e));
A(1,1)=a;
A(nx,nx)=h;
A(2,1)=d;
A(1,2)=b;
A(nx,nx-1)=m;
z=zeros(nx,nx);
for i=1:nx-2
    v(i)=g;
end
n=-2;
z=diag(v,n);
A1=plus(A,z);
B(1,1)=j*phi_i;
B(2,1)=k*phi_i;
B(nx,1)=l*phi_f;
B(3:nx-1,1)=0;
%% Matrix solver
inverse=inv(A1);
val=inverse*B;
%% Plot the values
x=[1:1:nx+2];
x(1,1)=0;
x(1,nx+2)=xl;
div=[dx/2:dx:xl-(dx/2)];
x(1,2:nx+1)=div;
yi=phi_i;
yf=phi_f;
y=[1:1:nx+2];
y(1,1)=yi;
y(1,nx+2)=yf;
y(1,2:nx+1)=val;
plot(x,y,'b','linewidth',1)
grid on
xlabel('Distance in metre');
ylabel('value of phi');
hold on
%% Exact solution
z=F/gamma; 
phi= phi_i+((phi_f-phi_i)*((exp(z*(div))-1))/(exp(z*xl)-1));
phi_val=[1:1:nx+2];
phi_val(1,1)=phi_i;
phi_val(1,nx+2)=phi_f;
phi_val(1,2:nx+1)=phi;
plot(x,phi_val,'r')
title('QUICK differencing scheme vs Exact solution');
legend('QUICK','Exact solution');
%% End of matlab programme
