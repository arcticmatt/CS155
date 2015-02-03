fileID_train = fopen('wine_training1.txt', 'r');
formatSpec = '%f %f %f %f %f %f %f %f %f %f %f %f';
sizeA = [12 inf];
raw_train = fscanf(fileID_train, formatSpec, sizeA);
x_train1 = raw_train(1:11,:);
y_train1 = raw_train(12,:);
x_train1 = transpose(x_train1);
y_train1 = transpose(y_train1);

fileID_train = fopen('wine_training2.txt', 'r');
formatSpec = '%f %f %f %f %f %f %f %f %f %f %f %f';
sizeA = [12 inf];
raw_train = fscanf(fileID_train, formatSpec, sizeA);
x_train2 = raw_train(1:11,:);
y_train2 = raw_train(12,:);
x_train2 = transpose(x_train2);
y_train2 = transpose(y_train2);

fileID_train = fopen('wine_training3.txt', 'r');
formatSpec = '%f %f %f %f %f %f %f %f %f %f %f %f';
sizeA = [12 inf];
raw_train = fscanf(fileID_train, formatSpec, sizeA);
x_train3 = raw_train(1:11,:);
y_train3 = raw_train(12,:);
x_train3 = transpose(x_train3);
y_train3 = transpose(y_train3);

fileID_train = fopen('wine_testing.txt', 'r');
formatSpec = '%f %f %f %f %f %f %f %f %f %f %f %f';
sizeA = [12 inf];
raw_test = fscanf(fileID_train, formatSpec, sizeA);
x_test = raw_test(1:11,:);
y_test = raw_test(12,:);
x_test = transpose(x_test);
y_test = transpose(y_test);

a = ridge(y_train1, x_train1, [.0001, 195.3125], 1);
c = inv(x_train1.'*x_train1+eye(11)*.0001)*x_train1.'*y_train1;

lambda = [.0001, .0005, .0025, .0125, .0625, .3125, 1.5625, 7.815, 39.0625, 195.3125];
x_trains{1} = x_train1;
x_trains{2} = x_train2;
x_trains{3} = x_train3;
y_trains{1} = y_train1;
y_trains{2} = y_train2;
y_trains{3} = y_train3;
e_ins = [];
e_outs = [];
norms = [];
for i = 1:3
    x_train = x_trains{i};
    y_train = y_trains{i};
    e_in = [];
    e_out = [];
    norm_list = [];
    for n = 1:10
        b = inv(x_train'*x_train+eye(11)*lambda(n))*x_train'*y_train;
        my_norm = norm(b);
        norm_list = [norm_list, my_norm];
        e_in = [e_in, (1/(size(x_train, 1)))*(norm(x_train*b - y_train))^2];
        e_out = [e_out, (1/(size(x_test, 1)))*(norm(x_test*b - y_test))^2];
    end
    e_ins{i} = e_in;
    e_outs{i} = e_out;
    norms{i} = norm_list;
end

data = ['-bo'; '-mo'; '-go'];
celldata = cellstr(data);
figure(1)
for i = 1:3
    semilogx(lambda, e_ins{i}, celldata{i} , 'MarkerSize', 5);
    hold on
end
xlabel('Lambda');
ylabel('E_{in}');
title('E_{in} vs Lambda');
l = legend('Wine Dataset 1', 'Wine Dataset 2', 'Wine Dataset 3');

figure(2)
for i = 1:3
    semilogx(lambda, e_outs{i}, celldata{i} , 'MarkerSize', 5);
    hold on
end
xlabel('Lambda');
ylabel('E_{out}');
title('E_{out} vs Lambda');
l = legend('Wine Dataset 1', 'Wine Dataset 2', 'Wine Dataset 3');

figure(3)
for i = 1:3
    semilogx(lambda, norms{i}, celldata{i} , 'MarkerSize', 5);
    hold on
end
xlabel('Lambda');
ylabel('Norm of w');
title('Norm of w vs Lambda');
l = legend('Wine Dataset 1', 'Wine Dataset 2', 'Wine Dataset 3');


