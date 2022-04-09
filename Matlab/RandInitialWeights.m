function W = RandInitialWeights(L_in, L_out)
    
    epsilon_init = sqrt(6)/(sqrt(L_in) + sqrt(L_out));
    W = -epsilon_init + rand(L_out, 1 + L_in) * 2 * epsilon_init;

end

