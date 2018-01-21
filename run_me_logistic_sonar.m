%The directory to be opened before running this is liblinear-2.1-->windows
addpath('../matlab');
clear all;
close all;
clc;

%load data
temp=load('sonarscale.mat');
scaled_data=temp.sonarscale(:,1:2:end);
train_data= load ('sonar-scale-train-indices.txt');
test_data= load ('sonar-scale-test-indices.txt');

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
model = train(y_training(31:150),sparse(training_data(31:150,:)),sprintf('-c %f -s 0',C(i)));
[predict_label, accuracy_va1, dec_values] = predict(y_training(1:30),sparse(training_data(1:30,:)), model);
[predict_label, accuracy_tr1, dec_values] = predict(y_training(31:150),sparse(training_data(31:150,:)), model);

set_2_x =removerows(training_data,'ind',[31:60]);
set_2_y =removerows(y_training,'ind',[31:60]);
model = train(set_2_y,sparse(set_2_x),sprintf('-c %f -s 0',C(i)));
[predict_label, accuracy_va2, dec_values] = predict(y_training(31:60),sparse(training_data(31:60,:)), model);
[predict_label, accuracy_tr2, dec_values] = predict(set_2_y,sparse(set_2_x), model);


set_3_x =removerows(training_data,'ind',[61:90]);
set_3_y =removerows(y_training,'ind',[61:90]);
model = train(set_3_y,sparse(set_3_x),sprintf('-c %f -s 0',C(i)));
[predict_label, accuracy_va3, dec_values] = predict(y_training(61:90),sparse(training_data(61:90,:)), model);
[predict_label, accuracy_tr3, dec_values] = predict(set_3_y,sparse(set_3_x), model);

set_4_x =removerows(training_data,'ind',[91:120]);
set_4_y =removerows(y_training,'ind',[91:120]);
model = train(set_4_y,sparse(set_4_x),sprintf('-c %f -s 0',C(i)));
[predict_label, accuracy_va4, dec_values] = predict(y_training(91:120),sparse(training_data(91:120,:)), model);
[predict_label, accuracy_tr4, dec_values] = predict(set_4_y,sparse(set_4_x), model);


set_5_x =removerows(training_data,'ind',[121:150]);
set_5_y =removerows(y_training,'ind',[121:150]);
model = train(set_5_y,sparse(set_5_x),sprintf('-c %f -s 0',C(i)));
[predict_label, accuracy_va5, dec_values] = predict(y_training(121:150),sparse(training_data(121:150,:)), model);
[predict_label, accuracy_tr5, dec_values] = predict(set_5_y,sparse(set_5_x), model);

sum_va = accuracy_va1(1)+accuracy_va2(1)+accuracy_va3(1)+accuracy_va4(1)+accuracy_va5(1);
sum_tr = accuracy_tr1(1)+accuracy_tr2(1)+accuracy_tr3(1)+accuracy_tr4(1)+accuracy_tr5(1);
%calculation average validation and training error for each C
err_va = [err_va (1-(sum_va*0.002))];
err_tr = [err_tr (1-(sum_tr*0.002))];
end
%Best C is the one with least validation error
[M,I]=min(err_va);

%Best C=1000. Evaluating testing data using this C
y_testing=testing_data(:,1);
testing_data=testing_data(:,2:end);
model = train(y_training,sparse(training_data),sprintf('-c %f -s 0',C(I)));
[predict_label, accuracy, dec_values] = predict(y_testing,sparse(testing_data), model);
test_error = 1-(accuracy(1)*0.002);

