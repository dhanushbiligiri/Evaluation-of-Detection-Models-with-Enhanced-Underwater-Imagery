clear all;
clc;

folder = 'images/Enhanced Val/images';
files = dir(fullfile(folder, '*.jpg'));
results = table([], [], 'VariableNames', {'Filename', 'Quality'});

for i = 1:length(files)
    filename = fullfile(folder, files(i).name);
    im = imread(filename); 
    quality = CCF(im); 
    results = [results; {files(i).name, quality}];
end

writetable(results, 'image_quality_resultsEnhancedval.csv');
