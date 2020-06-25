clc
clear all
close all
% ident


exp_file="./spab/exp_259.csv";
M = readtable(exp_file);
input = M(:,14);
input = input{:,:};
% 11 - node 7
output = M(:, 11);
output = output{:,:};

inputx = M(:,15:20);
inputx = inputx{:,:};
disp(inputx)

trim_start = 100;
input = input(trim_start:length(input),:);
output = output(trim_start:length(output),:);

% output = avg_outliers(output, 100);
subplot(211)
plot(output);
subplot(212)
plot(input);
% systemIdentification
