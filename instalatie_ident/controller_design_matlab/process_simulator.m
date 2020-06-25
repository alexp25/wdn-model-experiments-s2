clc
clear all
close all

Kp=3.18
Tp=0.5525
num=[Kp];
den=[Tp 1];
Hp=tf(num,den);
step(Hp)