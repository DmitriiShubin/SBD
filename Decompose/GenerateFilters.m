n_filters = 100;
fil_order = 300;

lpf = designfilt('lowpassfir','FilterOrder',fil_order ,'CutoffFrequency',1/n_filters,'DesignMethod','Window'); 
hpf = designfilt('highpassfir','FilterOrder',fil_order ,'CutoffFrequency',(n_filters-1)/n_filters,'DesignMethod','Window'); 

lpf = lpf.Coefficients;
hpf = hpf.Coefficients;

lpf = lpf';
hpf = hpf';

bpf = zeros(fil_order +1,n_filters-1);


for i=1:n_filters-1
    fil = designfilt('bandpassfir','FilterOrder',fil_order ,'CutoffFrequency1', (0.5+(i-1))/n_filters,'CutoffFrequency2',(1.5+(i-1))/n_filters,'DesignMethod','window'); 
    bpf(:,i) = fil.Coefficients;
end

bpf = [lpf,bpf];
bpf = [bpf,hpf];

lpf_dec = designfilt('lowpassfir','FilterOrder',fil_order ,'CutoffFrequency',0.05,'DesignMethod','Window'); 
lpf_dec = lpf_dec.Coefficients;

bpf = [bpf,lpf_dec'];

csvwrite('filters.csv',bpf)

