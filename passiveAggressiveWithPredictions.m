#Index Number:15020381
#Registration Number:2015/IS/038

#load the data
data=csvread("datafile.csv");

#replace 11th attribute - Class
data(data(:,11)==2,11) = -1;
data(data(:,11)==4,11) = 1;

#remove 1st attribute - Sample code number 
data(:,1) = [];

#input data
X = data(:,1:9);
Y = data(:,10);

#add bias
X = [X ones(size(X, 1), 1)];

#seperate training(2/3) and test data(1/3)
Xtrain=X(1:466,:);
Ytrain=Y(1:466,:);

Xtest = X(467:699,:);
Ytest = Y(467:699,:);

#prediction
file_id = fopen('prediction.txt', 'w');

#algorithm
for algorithmVariant = 1:3
  
  switch(algorithmVariant)
    case 1
      fprintf("Algorithm:PA\n");
      fprintf(file_id,"Algorithm:PA\n");
    case 2
      fprintf("Algorithm:PA-I\n");
      fprintf(file_id,"Algorithm:PA-I\n");
    case 3
      fprintf("Algorithm:PA-II\n");
      fprintf(file_id,"Algorithm:PA-II\n");
    otherwise
      fprintf('Invalid Algorithm Variant\n');
  endswitch
  #W = zeros(1,9);
  W=zeros(1,10);
  #iterations 1,2,10
  for iter = [1, 2, 10]
    for k = 1:iter
      #training for each algorithm variant
      for j = 1:466
        #aggressiveness parameter
        C = 1;
        x_t = Xtrain(j,:);
        y_t = Ytrain(j);
        #suffer loss
        l_t = max(0,1 - y_t*(W*x_t') );        
        switch (algorithmVariant)
          case 1
            tau_t = l_t/(norm(x_t)^2);
          case 2
            tau_t = min(C,l_t/(norm(x_t)^2));
          case 3
            tau_t = l_t/(((norm(x_t)^2))+1/(2*C));
          otherwise
            fprintf('Invalid Algorithm Variant\n');
        endswitch        
        W = W + tau_t*y_t*x_t;        
       end
     end
    
    #training accuracy
    correctlyClassifiedTrainingCount = 0;
    for j = 1:466
      x_t = Xtrain(j,:);
      y_t = Ytrain(j);
      y_pred = sign(W*x_t');
      if (y_t*y_pred == 1)
        correctlyClassifiedTrainingCount++;
      endif
    end
    trainingAccuracy = (correctlyClassifiedTrainingCount/466)*100;
        
    fprintf(file_id,"Number of iterations:%d\n",iter);    
    #test accuracy
    correctlyClassifiedTestcount = 0;
    for j = 1:233;
      x_t = Xtest(j,:);
      y_t = Ytest(j);
      y_pred = sign(W*x_t');
      if (y_t*y_pred == 1)
        fprintf(file_id,"%d is Correct\n",(466+j));
        correctlyClassifiedTestcount++;
      else
        fprintf(file_id,"%d is Incorrect\n",(466+j));      
      endif
    end
    
    testAccuracy = (correctlyClassifiedTestcount/233)*100;
   
    fprintf('Number of iterations:%d\n',iter);
    disp(sprintf('W: [%d, %d, %d, %d, %d, %d, %d, %d, %d]',W))
    fprintf('Training Accuracy:%d%%\n',trainingAccuracy);
    fprintf('Test Accuracy:%d%%\n',testAccuracy);
       
   end
   fprintf("\n");
   
end
