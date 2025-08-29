%% input: X is a rxc matrix
%% construct a sparse matrix Xout such that 
% for i=1:r/n
%    bigX((i-1)*n+1:i*n,(i-1)*c+1:i*c) = X((i-1)*n+1:i*n,:));
% end

function Xout = SURform3(X,n)
[r c] = size(X);
idi = kron((1:r)',ones(c,1));
idj =reshape(repmat(reshape(1:c*r/n,c,r/n),n,1),c*r,1);
Xout = sparse(idi,idj,reshape(X',r*c,1));