%% initialization
clear all;close all;
%clc; %#ok<CLALL>                                                          % cleaning the Workspace and Command Windows
file_name_1='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_1';     % the data file name
data_1 = importdata(strcat(file_name_1,'.txt'));                           
file_name_2='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_2'; 
data_2 = importdata(strcat(file_name_2,'.txt'));                           
file_name_3='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_3'; 
data_3 = importdata(strcat(file_name_3,'.txt'));                           
file_name_4='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_4'; 
data_4 = importdata(strcat(file_name_4,'.txt'));                           
file_name_5='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_5'; 
data_5 = importdata(strcat(file_name_5,'.txt'));                           
file_name_6='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_6'; 
data_6 = importdata(strcat(file_name_6,'.txt'));                          
file_name_7='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_7'; 
data_7 = importdata(strcat(file_name_7,'.txt'));                          
file_name_8='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_8'; 
data_8 = importdata(strcat(file_name_8,'.txt'));                            
file_name_9='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_9'; 
data_9 = importdata(strcat(file_name_9,'.txt'));                            
file_name_10='C:\Users\dario\Documents\Github\Valero_Foam_Hand\Data\Data_6sensors_0410\test_hand_single_0410_10'; 
data_10 = importdata(strcat(file_name_10,'.txt'));                          

data = [data_1;data_2;data_3;data_4;data_5;data_6;data_7;data_8;data_9;data_10];
donwsampling_rate = 50;                                                    % the downsampling rate
downsampled_data = downsample(data(25:end,:),donwsampling_rate);           % downsampling the data
donwsampled_and_labeled_data =...
    zeros(floor(size(data,1)/donwsampling_rate), size(data,2)+1);          % pre-alocating donwsampled_and_labeled_data
repeated_labels=repmat(1:6,1,ceil(size(downsampled_data,1)/6));            % generaqting labels
donwsampled_and_labeled_data(:,1:end-1)=downsampled_data;                  % downsampling the data
labels=repeated_labels(1:size(downsampled_data,1));                        % selecting labels consistent to the lenght of the recording
donwsampled_and_labeled_data(:,end)=labels;                                % assigning the labels to the downsampled data
%% plotting and savingthe processed data
figure(); plot(donwsampled_and_labeled_data(:,1:6))                        % plotting sensor readings
legend(...
    'Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6', 'location', 'southeast')  % setting the plot legend
csvwrite('donwsampled_and_labeled_data',donwsampled_and_labeled_data)                           % writing the preprocessed data into a csv file
%% KNN classification
rng(99)                                                                   % setting the random seed to a constant to have consistencu across results; can do a cross-validation for the final results
K=2;                                                                       % setting K in KNN; number of nearest neighbors to be participated in the voting
test_percentage=.8;                                                        % setting the test/train ratio
[train_data,test_data] = dividerand(donwsampled_and_labeled_data',...
    test_percentage, 1-test_percentage);                                   % randomized test, train split
mdl=fitcknn(train_data(1:6,:)',train_data(7,:)','NumNeighbors',K);         % fitting the KNN model
KNNPred=predict(mdl,test_data(1:6,:)');                                    % running the KNN model on the test data
accuracy=sum((KNNPred-test_data(end,:)')==0)/length(KNNPred);              % calculating the accuracy on the test set
disp(['accuracy is ', num2str(100*accuracy), '%'])                         % displaying accuracy percentage on the Command Window
%% plotting classification results
alpha_value = .5;                                                          % transparency of plot lines
plot(test_data(end,:)','color',[0, 0.4470, 0.7410, alpha_value],...
    'linewidth',3);hold on;                                                % plotting the test results
plot(KNNPred,'color',[0.8500, 0.3250, 0.0980, alpha_value],'linewidth',2); % plotting the test results
legend('test labels','predicted labels')                                   % test result legends
plotconfusion(categorical(test_data(end,:)'),categorical(KNNPred))         % plotting the confusion matrix
