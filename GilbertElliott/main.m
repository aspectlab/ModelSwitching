disp('Generating several gibabytes of data in data folder...')
datagen(0) % generate training data set
datagen(1) % generate validation data set
datagen(2) % generate test data set
disp('Computing IMM outputs...')
imm        % compute IMM outputs and performance
disp('Next steps: (1) run main_tcn.py in Python, and (2) run plotsresults.m in Matlab.')
