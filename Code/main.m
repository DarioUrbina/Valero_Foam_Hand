%% initialization
clear all;close all;
%clc; %#ok<CLALL>                                                          % cleaning the Workspace and Command Windows

ScriptDir = fileparts(mfilename('fullpath'));
MainDir = fileparts(ScriptDir);
donwsampling_rate = 50;  %determines the "step" of down sampling (downsampling_starting_point,+100,+200...)
downsampling_starting_point = 25; %determines which row to start from in the data matrix

%% Foam hand with 10 sensors

%turn "%{" into "%%{" to uncomment the chunk of code within the brackets
%{
file_name_1= append(MainDir,'\Data\Data_10sensors_foamhand_0416\test_hand_single_right 0416_5');     % the data file name
data = importdata(strcat(file_name_1,'.txt'));                %7000 values           
number_of_data_values = 7000;
number_of_labels = 8 ;
number_of_sensors = 10 ;
%}

%Latest Test
%%{
file_name_1= append(MainDir,'\Data\Data_10sensors_foamhand_0602\Data_Collection_5_extended');     % the data file name
file_name_2= append(MainDir,'\Data\Data_10sensors_foamhand_0602\Data_Collection_6_extended');     % the data file name
data = importdata(strcat(file_name_1,'.txt')); %1500
data_2 = importdata(strcat(file_name_2,'.txt')); %300
number_of_data_values = 1350;
number_of_labels = 6 ;
number_of_sensors = 10 ;
data = [data;data_2];
%}

%% Hand with 10 sensors

%{
 number_of_data_values = 3000;  
 number_of_labels = 5 ;      %5 hand positions
 number_of_sensors = 10 ;

 file_name_1= append(MainDir,'\Data\Data_10sensors_0411\test_hand_single_0411_1');
 data = importdata(strcat(file_name_1,'.txt'));
%}
 
%% hand with 6 sensors

%{     
 number_of_data_values = 3000;  %300 data values per file
 number_of_labels = 6 ;
 number_of_sensors = 6 ;
 
 file_name_1= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_1');
 data = importdata(strcat(file_name_1,'.txt'));         
 file_name_2= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_2') 
 data_2 = importdata(strcat(file_name_2,'.txt'));                           
 file_name_3= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_3') 
 data_3 = importdata(strcat(file_name_3,'.txt'));                           
 file_name_4= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_4') 
 data_4 = importdata(strcat(file_name_4,'.txt'));                           
 file_name_5= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_5') 
 data_5 = importdata(strcat(file_name_5,'.txt'));                           
 file_name_6= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_6') 
 data_6 = importdata(strcat(file_name_6,'.txt'));                          
 file_name_7= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_7')
 data_7 = importdata(strcat(file_name_7,'.txt'));                          
 file_name_8= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_8') 
 data_8 = importdata(strcat(file_name_8,'.txt'));                            
 file_name_9= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_9') 
 data_9 = importdata(strcat(file_name_9,'.txt'));                            
 file_name_10= append(MainDir,'\Data\Data_6sensors_0410\test_hand_single_0410_10') 
 data_10 = importdata(strcat(file_name_10,'.txt'));                          
%
 data = [data;data_2;data_3;data_4;data_5;data_6;data_7;data_8;data_9;data_10];

%}

%% Downsampling?

HFig1 = figure;
figure(HFig1); plot(data(1:number_of_data_values,1:number_of_sensors)) %Presents the undownsampled data in graph form

data = data(1:number_of_data_values,1:number_of_sensors); %Changes data into a data_values x number_of_sensors matrix
downsampled_data = downsample(data(downsampling_starting_point:end,:),donwsampling_rate);           % downsampling the data  decreases the sample rate of data by keeping the first sample and then every (downsampling_rate)th sample after the first. If data is a matrix, the function treats each column as a separate sequence  
downsampled_and_labeled_data =...
    zeros(floor(size(data,1)/donwsampling_rate), size(data,2)+1);          % pre-alocating donwsampled_and_labeled_data
                 %^gives number_of_data_values    ^gives number_of_sensors
     %^ gives zero matrix of size (#_of_data_vals/donwsampling_rate) x (num_of_sensors+1
repeated_labels=repmat(1:number_of_labels,1,ceil(size(downsampled_data,1)/number_of_labels));            % generating labels
                        %1x6 *[num_of_downsampled_values/number of labels]  (usually gives 1:number_of_labels)

downsampled_and_labeled_data(:,1:end-1)=downsampled_data;                  % downsampling the data
labels=repeated_labels(1:size(downsampled_data,1));                        % selecting labels consistent to the lenght of the recording
downsampled_and_labeled_data(:,end)=labels;                                % assigning the labels to the downsampled data
%% plotting and saving the processed data

%legend(...
 %   'Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6', 'location', 'southeast')  % setting the plot legend
csvwrite('downsampled_and_labeled_data',downsampled_and_labeled_data)                           % writing the preprocessed data into a csv file
HFig2 = figure;
figure(HFig2); plot(downsampled_and_labeled_data(1:number_of_labels,:))                         % plotting sensor readings
%doesn't show up for some reason. Put into console ^

%% KNN classification
rng(99)                                                                    % setting the random seed to a constant to have consistencu across results; can do a cross-validation for the final results
K=2;                                                                       % setting K in KNN; number of nearest neighbors to be participated in the voting
test_percentage=.8;                                                        % setting the test/train ratio
[train_data,test_data] = dividerand(downsampled_and_labeled_data',...
    test_percentage, 1-test_percentage);                                   % randomized test, train split

mdl=fitcknn(train_data(1:number_of_sensors,:)',train_data(number_of_sensors+1,:)','NumNeighbors',K);         % fitting the KNN model
KNNPred=predict(mdl,test_data(1:number_of_sensors,:)');                                    % running the KNN model on the test data

accuracy=sum((KNNPred-test_data(end,:)')==0)/length(KNNPred);              % calculating the accuracy on the test set
disp(['accuracy is ', num2str(100*accuracy), '%'])                         % displaying accuracy percentage on the Command Window
%% plotting classification results
alpha_value = .5;                                                          % transparency of plot lines

HFig3 = figure;
figure(HFig3)
plot(test_data(end,:)','color',[0, 0.4470, 0.7410, alpha_value],...
    'linewidth',3);hold on;                                                % plotting the test results
plot(KNNPred,'color',[0.8500, 0.3250, 0.0980, alpha_value],'linewidth',2); % plotting the test results
legend('test labels','predicted labels')                                   % test result legends

HFig4 = figure;
figure(HFig4)
plotconfusion(categorical(test_data(end,:)'),categorical(KNNPred))         % plotting the confusion matrix
%cm = confusionchart(categorical(test_data(end,:)),categorical(KNNPred));
%sortClasses(cm,["1","2","3","4","5"])