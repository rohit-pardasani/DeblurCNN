% Image blur function, reads a given image blurs it and save the result in
% give file
function J = blurImage(sourceFile,destFile)
    I = imread(sourceFile);
    h = ones(5,5)/25;
    J = imfilter(I,h);
    J = uint8(J);
    imwrite(J,destFile)
end

