i=3;
path = [RP_dir  '/../../Test_Pytorch/comparison_vehicle/5inits/init' num2str(i) ];
cd(path)
comparison_soft()
comparison_hard()
