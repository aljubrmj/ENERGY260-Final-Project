
%% Partition data into training, validation, and testing sets

load('CS229A_Dataset.mat'); 
total = 1:1330; %Set of all of the indices
Layers = 1; 
repeat = 1;
mse_train_matrix = zeros(length(Layers), repeat);
mse_test_matrix = zeros(length(Layers), repeat);
TestingPercentage_vec = 95:0.25:99.75; 
TrainingValidationPercentage_vec = 100 - TestingPercentage_vec; 

for i = 1:length(TestingPercentage_vec)
    
    TestingPercentage = TestingPercentage_vec(i);

    PercentTrainVal = (100-TestingPercentage)/100;
    PercentTest = TestingPercentage/100;
    NumberTrainRuns = round(PercentTrainVal*1330); %Number of Training Runs 
    NumberTestRuns = 1330-NumberTrainRuns; %Number of runs that will be tested 
    %tr = Partition(NumberTrainRuns, ICVrecord); %Latin Hypercube Sampling
    tr = randsample(1330, NumberTrainValidationRuns); %Random Sampling
    ts = setdiff(total,tr); %Finds the testing indices by subtracting total indices by training indices
    
    for j = 1:repeat

        [netOut, trOut] = ANNCrossValidationLoop(X', y', tr,ts, Layers); %Rows=features, Cols=dataPoints 
        m_matrix{i,j} = netOut;
        tr_matrix{i,j} = trOut;
        
        mse_train_matrix(i,j) = trOut.best_perf;
        mse_test_matrix(i,j) = trOut.best_tperf;
        
    end
    
end

mse_train_mean = mean(mse_train_matrix'); mse_train_std = std(mse_train_matrix');
mse_test_mean = mean(mse_test_matrix'); mse_test_std = std(mse_test_matrix');

plot(TrainingValidationPercentage_vec, mse_test_matrix); 
%figure(1); plot(TestingPercentage_vec, mse_test_mean)
%figure(2); plot(TestingPercentage_vec, mse_test_std)