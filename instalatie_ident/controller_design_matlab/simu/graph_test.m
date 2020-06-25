clc
clear all

% create graph

% use adjacency matrix
A = ones(4) - diag([1 1 1 1])

G = digraph(A~=0)