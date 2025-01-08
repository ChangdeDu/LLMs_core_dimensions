% function out = squareformq(in)
%
% Fast and general version of squareform that works also for matrices that
% are not distance matrices. Diagonal entries are ignored if they are
% present.

function out = squareformq(in)

inputdim = size(in);

% if we are dealing with a vector
if any(inputdim == 1)
    n = max(inputdim);
    sz = 0.5*(sqrt(8*n+1)+1);
    if sz~=round(sz), error('input cannot be a dissimilarity vector (too long or too short).'), end
    out = zeros(sz);
    out(tril(true(sz),-1)) = in; % lower diagonal
    out = out+out'; % fill upper diagonal
    
% if we are dealing with a square, symmetric matrix
elseif size(inputdim,2) == 2 && inputdim(1)==inputdim(2) && all(all(in)) == all(all(in'))
    
    out = in(tril(true(inputdim),-1));

% otherwise throw error
else  
    error('Input to squareformq must either be a vector or a 2D square, symmetric matrix.')
end