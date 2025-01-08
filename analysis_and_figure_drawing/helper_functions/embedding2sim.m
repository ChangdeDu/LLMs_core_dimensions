function cp = embedding2sim(embedding)

% This function converts an embedding to a similarity matrix

n_objects = size(embedding,1);

sim = embedding * embedding';
esim = exp(sim);
cp = zeros(n_objects,n_objects);
for i = 1:n_objects
    for j = i+1:n_objects
        ctmp = zeros(1,n_objects);
        for k = 1:n_objects
            if k == i || k == j, continue, end
            ctmp(k) = esim(i,j) / ( esim(i,j) + esim(i,k) + esim(j,k) );
        end
        cp(i,j) = sum(ctmp); % run sum first, divide all by 1852 later
    end
end
cp = cp/(n_objects-2); % complete the mean
cp = cp+cp'; % symmetric
cp(logical(eye(size(cp)))) = 1;