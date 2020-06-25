function [ index1, index2 ] = get_command_efficiency_index( u )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

index1 = std(u);
index2 = sqrt(sum(u.^2)/length(u));
% Average power = (1/N) * sum(|x(n)|.^2)

end

