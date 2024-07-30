


% Specify the directory path
folderPath = 'C:\Users\30678\Desktop\code\matlab\libs\matpower8.0\data';  % Replace with your folder path

% Get a list of all files in the folder
files = dir(folderPath);

for k = 1:length(files)
    [~, fileName, ~] = fileparts(files(k).name);
    
    % Check if fileName is not empty to avoid outputting directory names
    if ~isempty(fileName) && length(fileName)>4 && length(fileName)<11  && fileName(1:4)=="case"
        disp(fileName);
        mpc = loadcase(fileName);
        try
            [Bbus, BfO, Pbusinj, Pfinj] = makeBdc(mpc);%Bf*\theta+Pfinj
            BfO= full(BfO);
            Bf= BfO(:, 2: end);
            S = svd(Bf);
            S(end)
        catch
            continue
        end
    end
end