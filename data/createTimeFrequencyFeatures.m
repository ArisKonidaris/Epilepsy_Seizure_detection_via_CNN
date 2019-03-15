clear all
close all

%addpath('/home/vass/university/devel/tftb-0.2/mfiles')
addpath('C:\Users\Aris\Desktop\Epilepsy\code\tftb-0.2\mfiles')


%dirnames = {'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8'};

dirnames = {'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8'};


h=tftb_window(249, 'hamming'); % 50 ms
%h=tftb_window(249,'gauss'); % 50 ms


outDir = './data/sp/'
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

for i = 1:length(dirnames)
    filenames = dir(strcat('./data/', dirnames{i}, '/', '*.mat'));

    outDir = strcat('./data/sp/', dirnames{i});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    for j = 1:length(filenames)
        disp(filenames(j).name)
        fullpathname = strcat('./data/', dirnames{i}, '/', filenames(j).name);
        load(fullpathname);

        [numchannels, len] = size(data); 
        step = 39;

        sp_all = zeros(128, 128, numchannels);
        for c = 1:numchannels
            sp=tfrsp(squeeze(data(c,:))',1:step:len, 1024, h);
            %sp = tfrscalo(squeeze(data(c,:))', 1:step:len, 250, 0.05, 0.45, 128);
            %figure(c)
            %imagesc(sp(1:128, 2:end))
      
            sp_all(:, :, c) = sp(1:128, 2:end);
        end
        outFileName = strcat(outDir, filesep, filenames(j).name(1:end-3), 'sp');
          
        fileID = fopen(outFileName, 'w');
        fwrite(fileID, sp_all, 'float32');
        fclose(fileID);
    end
end

