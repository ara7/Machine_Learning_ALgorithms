%% Ara Lena , final submission
%The normalizing technique that gave best F1 score on test data is
%included, after that the code for Logistic regression, svm and
%mutinomial-dirichlet

%bow representaion
clear;
load('ml_challenge_author_attribution.mat');
for i =1 : length(train_txt)
bag(i,:) = histc(train_txt(i,:),1:10000);
end
for i =1:length(test_txt)
    bagTest(i,:) = histc(test_txt(i,:),1:10000);
end
%Most frequent 5000 words
freq = sum(bag);
[~, idx] = sort(freq, 'descend');
xtr1 = bag(:, idx(1:5000));

freqTest = sum(bagTest);
[~, idt] = sort(freqTest, 'descend');
xtes1 = bag(:, idt(1:5000));
bow=[xtr1;xtes1];

bow=bow-mean(bow); 
bowNew=bow./std(bow); 
xtrain_bow = bowNew(1:17153,:);
xtest_bow = bowNew(17154:end,:);

%% bow and nomalizing as code snippet provided by Dr. Dundar 
ind = [7501:10000]
bow_test=zeros(size(test_txt,1),length(ind));
bow_train=zeros(size(train_txt,1),length(ind));
parfor i=1:length(ind)
    bow_test(:,i)=sum(test_txt==ind(i),2);
    bow_train(:,i)=sum(train_txt==ind(i),2);
end
bow_test=bow_test./(sum(bow_test,2)*ones(1,length(ind)));
bow_train=bow_train./(sum(bow_train,2)*ones(1,length(ind)));

bow=[bow_train;bow_test];
for i=1:size(bow,2)
    bow(:,i)=(bow(:,i)-min(bow(:,i)))/max(bow(:,i)-min(bow(:,i)));
end
bow_train1=bow(1:size(bow_train,1),:);
bow_test1=bow(size(bow_train,1)+1:size(bow_train,1)+size(bow_test,1),:);
xtrain = bow_train1; xtest = bow_test1;

%% 1. Logistic regression
m = size(xtrain, 1);
n = size(xtrain, 2);
num_labels = 45;
lambda = 0.1;
all_theta = zeros(num_labels, n + 1);

xtrain = [ones(m, 1) xtrain]; %bias term


for c = 1:num_labels
 initial_theta = zeros(n+1,1);
 options = optimset('GradObj', 'on', 'MaxIter', 50);
 all_theta(c,:) = fmincg(@(t)(costFunction(t,xtrain,(train_author==c),lambda)), ...
  initial_theta,options);
end 

mTest = size(xtest,1);
nTest = size(xtest,2);
xtest = [ones(mTest,1) xtest]; %bias term
h = sigmoid(all_theta * xtest')
[M,I]=max(h);

prediction=I;


%% 2. SVM
s = 1;
c=0.1;
option = ['-s ' num2str(s) ' -c ' num2str(c) ' -B 1 -q'];
model_submission = train(ytrain, sparse(xtrain), option);
svm_predict = predict([1:12728]', sparse(xtes), model_submission,'-q');

%% 3. Multi-dirichlet

alpha = 0.5;      

xtrain1 = sparse(xtrain); %or  use sparse matrix
xtest1 = sparse(xtest);

for i=1:max(train_author)% Training
Nk(i,:)=sum(xtrain1(train_author==i,:),1);
end
N=sum(Nk,2);       
M=sum(xtest1,2);    
[n d]=size(xtest1); 
for i=1:max(train_author)% likelihoods
alpha0=d*alpha;    
A=gammaln(N(i)+alpha0)-gammaln(M+N(i)+alpha0); 
den=gammaln(Nk(i,:)+alpha); 
num=gammaln(xtest1+ones(n,1)*(Nk(i,:)+alpha)); 
loglik(:,i)=sum(num-ones(n,1)*den,2)+A;
end

[val pred]=max(loglik,[],2);
labels = pred;




