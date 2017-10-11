function [] = write_cell_contents(filename, cell_array)
%   Writes a cell array of strings to a file
    fid=fopen(filename,'wt');
    csvFun = @(str)sprintf('%s\\n',str);
    array_str = cellfun(csvFun, cell_array, 'UniformOutput', false);
    array_str = strcat(array_str{:});
    fprintf(fid,array_str);
    fclose(fid);
end

