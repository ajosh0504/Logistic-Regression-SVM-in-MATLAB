%The directory to be opened before running this is liblinear-2.1-->windows
addpath('../matlab');
clear all;
close all;
clc;

%load data
temp=load('breastcancerscale.mat');
scaled_data=temp.breastcancerscale(:,1:2:end);
train_data= load ('breast-cancer-scale-train-indices.txt');
test_data= load ('breast-cancer-scale-test-indices.txt');

%Build training and testing sets
testing_data =removerows(scaled_data,'ind',train_data);
training_data =removerows(scaled_data,'ind',test_data);
%cross validation
y_training=training_data(:,1);
training_data=training_data(:,2:end);
C=[0.1 1 10 100 1000];
err_va = [];
err_tr=[];

%calculating testing and validation error for each c.
%accuracy_tr are the training accuracies and accuracy_va are the validation
%accuracies
for i=1:5
model = train(y_training(101:500),sparse(training_data(101:500,:)),sprintf('-c %f -s 3',C(i)));
[predict_label, accuracy_va1, dec_values] = predict(y_training(1:100),sparse(training_data(1:100,:)), model);
[predict_label, accuracy_tr1, dec_values] = predict(y_training(101:500),sparse(training_data(101:500,:)), model);

set_2_x =removerows(training_data,'ind',[101:200]);
set_2_y =removerows(y_training,'ind',[101:200]);
model = train(set_2_y,sparse(set_2_x),sprintf('-c %f -s 3',C(i)));
[predict_label, accuracy_va2, dec_values] = predict(y_training(101:200),sparse(training_data(101:200,:)), model);
[predict_label, accuracy_tr2, dec_values] = predict(set_2_y,sparse(set_2_x), model);


set_3_x =removerows(training_data,'ind',[201:300]);
set_3_y =removerows(y_training,'ind',[201:300]);
model = train(set_3_y,sparse(set_3_x),sprintf('-c %f -s 3',C(i)));
[predict_label, accuracy_va3, dec_values] = predict(y_training(201:300),sparse(training_data(201:300,:)), model);
[predict_label, accuracy_tr3, dec_values] = predict(set_3_y,sparse(set_3_x), model);

set_4_x =removerows(training_data,'ind',[301:400]);
set_4_y =removerows(y_training,'ind',[301:400]);
model = train(set_4_y,sparse(set_4_x),sprintf('-c %f -s 3',C(i)));
[predict_label, accuracy_va4, dec_values] = predict(y_training(301:400),sparse(training_data(301:400,:)), model);
[predict_label, accuracy_tr4, dec_values] = predict(set_4_y,sparse(set_4_x), model);


set_5_x =removerows(training_data,'ind',[401:500]);
set_5_y =removerows(y_training,'ind',[401:500]);
model = train(set_5_y,sparse(set_5_x),sprintf('-c %f -s 3',C(i)));
[predict_label, accuracy_va5, dec_values] = predict(y_training(401:500),sparse(training_data(401:500,:)), model);
[predict_label, accuracy_tr5, dec_values] = predict(set_5_y,sparse(set_5_x), model);

sum_va = accuracy_va1(1)+accuracy_va2(1)+accuracy_va3(1)+accuracy_va4(1)+accuracy_va5(1);
sum_tr = accuracy_tr1(1)+accuracy_tr2(1)+accuracy_tr3(1)+accuracy_tr4(1)+accuracy_tr5(1);
%calculation average validation and training error for each C
err_va = [err_va (1-(sum_va*0.002))];
err_tr = [err_tr (1-(sum_tr*0.002))];
end
%Best C is the one with least validation error
[M,I]=min(err_va);

%Best C=10. Evaluating testing data using this C
y_testing=testing_data(:,1);
testing_data=testing_data(:,2:end);
model = train(y_training,sparse(training_data),sprintf('-c %f -s 3',C(I)));
[predict_label, accuracy, dec_values] = predict(y_testing,sparse(testing_data), model);
test_error = 1-(accuracy(1)*0.002);

