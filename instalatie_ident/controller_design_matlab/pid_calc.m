clc
clear all

s=tf('s');

Kp=0.15;
Tp=1.53;
K0=1;
T0=1;

Hp=Kp/(Tp*s+1)


H0=K0/(T0*s+1)

Hd = H0/(1-H0);
Hr = Hd/Hp

minreal(Hr)