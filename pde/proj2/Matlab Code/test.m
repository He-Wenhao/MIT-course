% Define the function f(i) as a separate function or inline
f = @(i) i^2;  % Example function, replace with your own function

% Initialize parameters
N = 10;             % Length of the array
A = zeros(1, N);    % Initialize a row vector of N zeros

% Apply the function to each element
for i = 1:N
    A(i) = f(i);    % Call the function f with the current index i
end

% Display the updated array
disp(A);
