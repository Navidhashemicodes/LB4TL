%%%% case 1
clear all
close all
clc
%%%% Assuming that python is installed and recognized by os
py_dir = 'python';
RP_dir = pwd;
addpath(genpath( [RP_dir '/../../src'] ) );
path = [RP_dir  '/../../Test_Pytorch/Georgios_vehicle_navigation_Pytorch'];
cd(path)
comparison_soft()
comparison_hard()
cd ..
cd ..
cd(RP_dir)
