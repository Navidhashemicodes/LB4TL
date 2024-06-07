%%%% Table
clear all
close all
clc


py_dir = 'python';
RP_dir = pwd;
addpath(genpath( [RP_dir '/../../src'] ) );

value = cell(5,4);

figure()

i=1;
path = [RP_dir  '/../../Test_Pytorch/comparison_vehicle/5inits/init' num2str(i) ];
cd(path)
comparison_soft()
comparison_hard()

