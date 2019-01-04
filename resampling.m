abnorm_new = zeros(size(norm,1),129);

buff = zeros(1,187);

for i=1:size(abnorm,1)
    buff = abnorm(i,1:187);
    abnorm_new(i,129) = abnorm(i,188);
    abnorm_new(i,1:128) = resample(buff',85,125);
end

for i=1:size(norm,1)
    buff = norm(i,1:187);
    norm_new(i,129) = norm(i,188);
    norm_new(i,1:128) = resample(buff',85,125);
end

data = [norm; abnorm];

csvwrite('data.csv',data)
