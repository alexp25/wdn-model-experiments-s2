clc
clear all
close all
% ident

index = 8;

exp_file_input=strcat("./spab/ident_pump_prbs_2_exp_259_", num2str(index), ".csv");
disp(exp_file_input)
exp_file_output=strcat("./spab/ident_flow_prbs_2_exp_259_", num2str(index), ".csv");
disp(exp_file_output)

M = readtable(exp_file_input);
input = M(:,1);
input = input{:,:};

M2 = readtable(exp_file_output);
output = M2(:,1);
output = output{:,:};

trim_start = 200;
input = input(trim_start:length(input),:);
output = output(trim_start:length(output),:);

output = avg_outliers(output, 1000);
subplot(211)
plot(output);
subplot(212)
plot(input);
% systemIdentification
