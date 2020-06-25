clc
clear all
close all

% global
d=0.02;
g=9.8;
r=d/2;
kp=0.5;
% kp=1;
mu=0.001; % viscosity, Pascal * sec

% L/h
F0_pump=600;
F0_pump=F0_pump/3600/1000;

% % H1 - conducta lunga
% kp=0.5;
% F0=F0_pump;
% L=10;
% V0=pi*r^2*L;
% k=8*mu*L/(pi*r^4);
% S=pi*r^2;
% tp=L^5/(2*k*F0*S);
% H1n=[kp]
% H1d=[tp 1]


F0=F0_pump;
L=1;
V0=pi*r^2*L;
% http://www.termo.utcluj.ro/mf/luc8.pdf
% alfa=4*F0/(pi*d^2*sqrt(2*g*L));
alfa=1;
tp=alfa^2*V0/F0;
H1n=[kp]
H1d=[tp 1]


% H2 - conducta scurta
F0=F0_pump/2;
L=0.2;
V0=pi*r^2*L;
% http://www.termo.utcluj.ro/mf/luc8.pdf
% alfa=4*F0/(pi*d^2*sqrt(2*g*L));
alfa=1;
tp=alfa^2*V0/F0;
H2n=[kp]
H2d=[tp 1]

% H3 - conducta scurta
F0=F0_pump/2;
L=0.2;
V0=pi*r^2*L;
% http://www.termo.utcluj.ro/mf/luc8.pdf
% alfa=4*F0/(pi*d^2*sqrt(2*g*L));
alfa=1;
tp=alfa^2*V0/F0;
H3n=[kp]
H3d=[tp 1]
% 
% % H4 - conducta lunga
% kp=0.5;
% F0=F0_pump;
% L=10;
% V0=pi*r^2*L;
% k=8*mu*L/(pi*r^4);
% S=pi*r^2;
% tp=L^5/(2*k*F0*S);
% H4n=[kp]
% H4d=[tp 1]

F0=F0_pump;
L=1;
V0=pi*r^2*L;
% http://www.termo.utcluj.ro/mf/luc8.pdf
% alfa=4*F0/(pi*d^2*sqrt(2*g*L));
alfa=1;
tp=alfa^2*V0/F0;
H4n=[kp]
H4d=[tp 1]


% s1=tf(H1n,H1d);
% step(s1);
% 
% figure 
% s4=tf(H4n,H4d);
% step(s4);


Kp=10;
Ti=1;
Td=0.1;
Cn=[Kp*Ti*Td Ti 1]
Cd=[0.01 Ti 0]