options(warn=-1)
source("./bayesian_approach/bayes.R");

# Taxa in cols
# Obs in rows
# Header taxa
for(j in 0:9){
  for(i in 1:7){
    file = paste('./data/', 'Dissimilarity/5/', 'P', i, j, '.csv', sep='', collapse='')
    dt <- prepare_data(file)
    ex = dt$community %>% unique %>% sample(9)
    stan_results <- fit_stan(dt, stan_file = "./code/bayesian_approach/Stan_file.stan",
                      exclude = ex, chains = 2, cores = 2,
                      iter = 15000, warmup = 7500, thin = 15, seed=12)

    br <- bootstrap_results(stan_results, nboot=10)

    tst = getPredictions(br)
    trn = getObservations(br)

    name = paste(c('./results/Maynard/trn',i,j,'.csv'), collapse = '')
    write.csv(trn, name, row.names = FALSE)

    name = paste(c('./results/Maynard/tst',i,j,'.csv'), collapse = '')
    write.csv(tst, name, row.names = FALSE)

    plot_boot_results(br,"obs_pred")
  }
}
