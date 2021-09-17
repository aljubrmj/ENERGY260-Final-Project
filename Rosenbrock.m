%% Impact of Latin Hypercube Sampling In Creating Proxy Functions
clear
clc
close all

%Define and plot Rosenbrock function
f = @(x,y) (1-x).^2 + 100*(y-x.^2).^2;
figure (1); fsurf(f)
title('Rosenbrock Function'); 
xlabel('x'); 
ylabel('y'); 
zlabel('f(x,y)'); 


%Fix number of samples from to be drawn to
N = 50:50:500; 

%Mean-square error evaluation range
Range = -5:0.1:5;

%Initiate empty arrays to be filled up
x_Random = []; y_Random = []; 
x_Lhc = []; y_Lhc = []; 
MSE_Random = []; MSE_Lhc = []; 

for i = N
    %% Random Sampling

    %Draw random samples between [-5,5]
    a = -5;
    b = 5;
    x = (b-a).*rand(i,1) + a;
    y = (b-a).*rand(i,1) + a;
    x_Random = [x_Random; x]; 
    y_Random = [y_Random; y]; 
    z_Random = f(x_Random,y_Random);

    %Fit Gaussian SVM model on the randomly sampled data and plot accordingly
    rng default
    MdlRandom = fitrsvm([x_Random,y_Random], z_Random, ...
            'Verbose',1 , 'Standardize', 1, 'KernelFunction', 'gaussian');

    %PRandom = @(x,y) predict(MdlRandom, [x,y]);
    %figure(2); fsurf(PRandom)

    %Calculate Mean Square Error of both model at a range of values to compare

    MSE_Random = [MSE_Random immse(f(Range, Range), predict(MdlRandom, [Range', Range'])')]; 
    
    %% Latin Hypercube Sampling
    %Draw Latin Hypercube Samples between [-5, 5]
    [LhcSample,~]=lhs(i,[-5 -5],[5 5]);
    x = LhcSample(:,1); 
    y = LhcSample(:,2);
    x_Lhc = [x_Lhc; x]; 
    y_Lhc = [y_Lhc; y]; 
    z_Lhc = f(x_Lhc,y_Lhc);

    %Fit Gaussian SVM model on the randomly sampled data and plot accordingly
    rng default
    MdlLhc = fitrsvm([x_Lhc,y_Lhc], z_Lhc, ...
            'Verbose',1 , 'Standardize', 1, 'KernelFunction', 'gaussian');

    %PLhc = @(x,y) predict(MdlLhc, [x,y]);
    %figure(3); fsurf(PLhc)

    %Calculate Mean Square Error of both model at a range of values to compare
    MSE_Lhc = [MSE_Lhc immse(f(Range, Range), predict(MdlLhc, [Range', Range'])')]; 
end 

%Plot MSE of Random and Lhc sampling techniques 
figure(2), semilogy(N, MSE_Random, N, MSE_Lhc)
title('Mean-Square Error of Random and Latin Hypercube Sampling Techniques');
xlabel('Number of Samples'); 
ylabel('MSE'); 
legend('Random Sampling', 'Latin Hypercube Sampling'); 