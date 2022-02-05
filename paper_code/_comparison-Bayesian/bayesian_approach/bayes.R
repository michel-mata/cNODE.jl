
library(rstan)
library(coda)
library(tidyverse)

prepare_data <- function(filename = "../../data/Kuebbing_plants/natives.csv"){
	# Input
	# filename: file name of a csv file containing the endpoints (rows are observations, columns are species, headers are species names)
	# Output
	# a tibble as the csv file but with an extra column containing the label for the community
	dt <- read_csv( filename , col_names = FALSE,
				col_types=cols(
  					X1 = col_double(),
  					X2 = col_double(),
  					X3 = col_double(),
  					X4 = col_double(),
  					X5 = col_double())
					)
	rownames(dt) = NULL
	# now label communities
	spnames <- colnames(dt)
	dt <- dt %>% add_column(community = apply(dt, 1, function(x) paste0(spnames[x > 0], collapse = "-")))
	return(dt)
}

# estimate B using a Bayesian appraoch via Stan
fit_stan <- function(dt, stan_file, exclude = NULL, B_upper, B_lower,
					 chains, cores, iter, thin, warmup, seed=10, delta=0.85, treedepth=12){
	# make sure it has a names() field present
	if(!is_tibble(dt) | !is.data.frame(dt)){
		stop("E must be a tibble or data frame")
	}
	# strip out the exluded communities if provided
	if(!is.null(exclude)){
		# make sure the comm labels are in the right order
		exclude <- as.character(sapply(exclude, function(x) paste(sort(strsplit(x,"-")[[1]]),collapse="-")))
		# make sure the excluded communities are labeled right
		if(any(!exclude%in%unique(dt$community))){
			stop("`exclude' contains unobserved communities")
		}
		else{
			print(paste("excluding communities",paste(exclude, collapse = ", ")))
			# remove the excluded communities and the community indicator
			E <- dt %>% filter(!community%in%exclude) %>% select(-community) %>% as.matrix()
			tst = dt %>% filter(community%in%exclude) %>% select(-community) %>% as.matrix()
		}
	}
	else{
		print("fitting all endpoints")
		E <- dt %>% select(-community) %>% as.matrix()
	}
	nspp <- ncol(E)
	sp_names <- colnames(E)
	# create upper/lower bounds if not supplied
	if(missing(B_upper)){
		# constrain the community to be competitive (zero is the upper bounds for each B_ij)
		B_upper <- matrix(0,nspp,nspp)
	}
	if(missing(B_lower)){
		# no lower bound
		B_lower <- matrix(-1,nspp,nspp)
	}
	# get the intial B matrix, taking the max obs values, assuming a diagonal matrix
	B_init <- -diag(1/apply(dt %>% select(-community) %>% as.matrix(), 2, function(x) quantile(x[x>0],0.9)))
	B_init[B_init==0] <- -1e-10
	# set the s.d. for the prior to be the largest diagonal element
	maxB <- max(abs(B_init))
	# calcualte the empirical log sd to initialize the mcmc chains
	sigmax_init <- dt %>% gather(species,abundance,-community) %>% filter(abundance>0) %>%
							group_by(species, community) %>% summarize(sd = sd(log(abundance))) %>% ungroup %>% filter(complete.cases(.)) %>%
							group_by(species) %>% summarize(med_sd = mean(sd)) %>% arrange(species) %>% select(med_sd) %>% unlist %>% as.numeric
	# build stan data list
	stan_data <- list(	N = ncol(E),
						M = nrow(E),
						E = E,
						B_upper = B_upper,
						B_lower = B_lower,
						y = rep(-1,nspp),
						maxB = maxB)
	# initialization vector
	init_list <- replicate(chains,list(sigmax = sigmax_init, B = B_init), simplify = FALSE)
	# stan options
	options(mc.cores = parallel::detectCores())
	Sys.setenv(USE_CXX14 = 1)
	rstan_options(auto_write = TRUE)
	# check to make sure we have enough endpoints
	check_endpoints(E)
	# fit the stan model!
	stan_fit <- stan(stan_file, data=stan_data, cores = cores, iter=iter, thin = thin, warmup=warmup,
						chains = chains, seed=seed, init=init_list, control = list(adapt_delta = delta, max_treedepth=treedepth))
	return(list(E_full = dt, E = E, stan_fit=stan_fit, exclude = exclude, B_upper = B_upper, B_lower = B_lower, B_init = B_init, seed = seed, trn=E,tst=tst))
}

# make sure we have enough endpoints, using lm. Returns an error code if not enough
check_endpoints <- function(E){
	nspp <- ncol(E)
	# get the median value across
	Emean <- E %>% as_tibble %>% add_column(community = apply(., 1, function(x) paste0(colnames(.)[x > 0], collapse = "-"))) %>%
		gather(species, abundance, -community) %>% filter(abundance>0) %>% group_by(species, community) %>%
		summarize(abundance = median(abundance)) %>% spread(species, abundance, fill=0) %>% select(-community) %>% as.matrix()
	B_lm <- matrix(NA,nspp,nspp)
	# fit each species independently, filling in each row of B
	for(i in 1:nspp){
		myE <- Emean[Emean[,i]>0,]
		B_lm[i,] <- as.numeric(lm(rep(-1,nrow(myE))~-1+myE)$coefficients)
	}
	if(any(is.na(B_lm))){
		stop("Not enough endpoints to fit")
	}
}

# plot the mcmc results, along with histograms and obs vs. pred
plot_diagnostics <- function(stan_results, show_plot="both"){
	if(show_plot == "both"){
		show_plot <- c("chains","hist")
	}
	if(!any(c("chains","hist")%in%show_plot)){
		stop("Invalid plot option, 'show_plot' must be 'chains', 'hist', or 'both'")
	}
	with(stan_results, {
		sp_names <- colnames(E)
		nspp <- length(sp_names)
		if("chains"%in%show_plot){
			print("Plotting MCMC chains")
			# plot the mcmc runs
			for(i in 1:length(stan_fit@model_pars)){
				if(stan_fit@model_pars[i]!="Blim"){
					post_plot<-As.mcmc.list(stan_fit,pars=stan_fit@model_pars[i]) # this samples from the posterior
					if(ncol(post_plot[[1]])>1){
							show(plot(post_plot)) # this shows the convergence
					}
					else{
						show(plot(post_plot,main=stan_fit@model_pars[i]))
					}
				}
			}
		}
		# wrap the chains into a matrix
		stan_fit <- as.matrix(stan_fit)
		# grab the individual components and rename
		B_stan <- stan_fit[,1:nspp^2] %>% as_tibble() %>% setNames(apply(expand.grid(sp_names,sp_names),1,function(x) paste(x,collapse="-")))
		if("hist"%in%show_plot){
			print("Plotting histogram of B")
			# plot the histograms
			show(ggplot(B_stan %>% gather("B_ij", "value"), aes(x = value)) + geom_histogram() + geom_vline(xintercept=0, col="darkred", linetype=2)+facet_wrap(~B_ij, scales = "free") +
						 	theme(axis.text.x = element_text(size=5))+theme_bw())
		}
	})

}

# boostrap the mcmc results and calculate predicted abundances and probability of coexistence
bootstrap_results <- function(stan_results, nboot){
	with(stan_results, {
		nspp <- ncol(E)
		sp_names <- colnames(E)
		# conver the stan results to a matrix
		stan_mat <- as.matrix(stan_fit)
		# extract the B entries
		B_stan <- stan_mat[,1:nspp^2]
		# extract the sigma entries
		sig_stan <- stan_mat[,paste0("sigmax[",1:nspp,"]")]


		# create a matrix of communities presence/absence and label name
		labels <- expand.grid(replicate(nspp, 0:1, simplify = FALSE)) %>% filter(rowSums(.)>0) %>% as_tibble %>% setNames(sp_names) %>%
			mutate(community = as.numeric((.>0)%*%(2^(0:(ncol(.)-1))))) %>% gather(species,abundance,-community) %>% filter(abundance>0) %>%
			select(-abundance) %>% group_by(community) %>% summarize(label=paste(species,collapse="-")) %>% left_join(expand.grid(replicate(nspp, 0:1, simplify = FALSE)) %>% filter(rowSums(.)>0) %>% as_tibble %>% setNames(sp_names) %>%
			mutate(community = as.numeric((.>0)%*%(2^(0:(ncol(.)-1))))), "community") %>% select(-community) %>% rename(community = label)
		# if doing out of fit, only predict the unobserved
		if(any(exclude%in%labels$community)){
			labelsQ <- labels %>% filter(community%in%exclude)
		}

		abund_Q <- coexist_Q <- tibble()
		# Q
		for(i in 1:nrow(labelsQ)){
			#print(paste("Bootstrapping community",i,"of", nrow(labelsQ)))
			# get the current community id and species locations
			my_id <- labelsQ$community[i]
			my_sp <- (1:nspp)[labelsQ %>% select(-community) %>% slice(i) %>% unlist %>% as.logical()]
			n_coex <- 0
			for(j in 1:nboot){
				# sample an index
				ind <- sample(1:nrow(B_stan),1)
				# get the B matrix
				my_B <- matrix(B_stan[ind,],nspp,nspp)[my_sp, my_sp]
				# get the sigmas
				my_sig <- as.numeric(sig_stan[ind,my_sp])
				# calculate the solution
				my_x <- -rowSums(solve(my_B))
				my_x_noise <- rep(NA, length(my_x))
				if(all(my_x>0)){
					# if all species are present, sample from lognormal with sigma
					my_x_noise <- as.numeric(apply(cbind(my_x, my_sig), 1, function(x) rlnorm(1, log(x[1]), x[2])))
					# add one to the coexistence counter
					n_coex <- n_coex+1
				}
				# add to abund_mat
				abund_Q <- bind_rows(abund_Q,tibble(community = my_id, species = sp_names[my_sp], abundance = my_x, abundance_noise = my_x_noise, coexist = as.numeric(all(my_x>0)), out_fit = as.numeric(my_id%in%exclude)))
			}

			# get the proportion that go extinct
			coexist_Q <- bind_rows(coexist_Q, tibble(community = my_id, num_coexist = n_coex, num_comms = nboot) %>% mutate(prop_extinct = (num_comms - n_coex)/num_comms))
		}

		# if doing out of fit, only predict the unobserved
		if(any(exclude%in%labels$community)){
			labelsP <- labels %>% filter(!community%in%exclude)
		}

		abund_P <- coexist_P <- tibble()
		# P
		for(i in 1:nrow(labelsP)){
			#print(paste("Bootstrapping community",i,"of", nrow(labelsP)))
			# get the current community id and species locations
			my_id <- labelsP$community[i]
			my_sp <- (1:nspp)[labelsP %>% select(-community) %>% slice(i) %>% unlist %>% as.logical()]
			n_coex <- 0
			for(j in 1:nboot){
				# sample an index
				ind <- sample(1:nrow(B_stan),1)
				# get the B matrix
				my_B <- matrix(B_stan[ind,],nspp,nspp)[my_sp, my_sp]
				# get the sigmas
				my_sig <- as.numeric(sig_stan[ind,my_sp])
				# calculate the solution
				my_x <- -rowSums(solve(my_B))
				my_x_noise <- rep(NA, length(my_x))
				if(all(my_x>0)){
					# if all species are present, sample from lognormal with sigma
					my_x_noise <- as.numeric(apply(cbind(my_x, my_sig), 1, function(x) rlnorm(1, log(x[1]), x[2])))
					# add one to the coexistence counter
					n_coex <- n_coex+1
				}
				# add to abund_mat
				abund_P <- bind_rows(abund_P,tibble(community = my_id, species = sp_names[my_sp], abundance = my_x, abundance_noise = my_x_noise, coexist = as.numeric(all(my_x>0)), out_fit = as.numeric(my_id%in%exclude)))
			}

			# get the proportion that go extinct
			coexist_P <- bind_rows(coexist_P, tibble(community = my_id, num_coexist = n_coex, num_comms = nboot) %>% mutate(prop_extinct = (num_comms - n_coex)/num_comms))
		}

		return(append(stan_results, list(boot_X = abund_Q, boot_C = coexist_Q, P=abund_P, Q=abund_Q)))
	})
}

getPredictions <- function(br){
      Q = br$Q %>% mutate(community = factor(community, level=br$Q %>% select(community) %>% distinct() %>% mutate(len = nchar(community)) %>% arrange(len, community) %>% select(-len) %>% unlist %>% as.character()))
      obs_X_filt = br$E_full %>% filter(community%in%unique(Q$community)) %>% gather(species, abundance, -community)
      obs_X = obs_X_filt %>% mutate(community = factor(community, level	= br$E_full %>% select(community) %>% distinct() %>% mutate(len = nchar(community)) %>% arrange(len, community) %>% select(-len) %>% unlist %>% as.character())) %>% filter(abundance>0)
      obs_X = obs_X_filt %>% group_by(community, species) %>% filter(abundance>0) %>% summarize(Observed = median(abundance)) %>% left_join( Q %>% mutate(community = as.character(community)) %>% filter(coexist==1, abundance>0) %>% group_by(community, species) %>% summarize(Predicted = median(abundance)), c("community","species"))
    }

getObservations <- function(br){
      P = br$P %>% mutate(community = factor(community, level=br$P %>% select(community) %>% distinct() %>% mutate(len = nchar(community)) %>% arrange(len, community) %>% select(-len) %>% unlist %>% as.character()))
      obs_X_filt = br$E_full %>% filter(community%in%unique(P$community)) %>% gather(species, abundance, -community)
      obs_X = obs_X_filt %>% mutate(community = factor(community, level	= br$E_full %>% select(community) %>% distinct() %>% mutate(len = nchar(community)) %>% arrange(len, community) %>% select(-len) %>% unlist %>% as.character())) %>% filter(abundance>0)
      obs_X = obs_X_filt %>% group_by(community, species) %>% filter(abundance>0) %>% summarize(Observed = median(abundance)) %>% left_join( P %>% mutate(community = as.character(community)) %>% filter(coexist==1, abundance>0) %>% group_by(community, species) %>% summarize(Predicted = median(abundance)), c("community","species"))
    }

# plot the boostrap results, both violin and obs vs. pred
plot_boot_results <- function(boot_results, show_plot = "both", return_plots = FALSE){
	if(show_plot == "both"){
		show_plot <- c("violin","obs_pred")
	}
	if(!any(c("violin","obs_pred")%in%show_plot)){
		stop("Invalid plot option, 'show_plot' must be 'violin' or 'obs_pred', or 'both'")
	}
	out_list <- NULL
	with(boot_results, {
		# relevel the communitys by size
		boot_X <- boot_X %>% mutate(community = factor(community, level=boot_X %>% select(community) %>% distinct() %>% mutate(len = nchar(community)) %>% arrange(len, community) %>% select(-len) %>% unlist %>% as.character()))
		# gather the observed datapoints, filter to the current fitted values, and relevel community
		obs_X_filt <- E_full %>% filter(community%in%unique(boot_X$community)) %>% gather(species, abundance, -community)
		obs_X <- obs_X_filt %>% mutate(community = factor(community, level	= E_full %>% select(community) %>% distinct() %>% mutate(len = nchar(community)) %>%
				arrange(len, community) %>% select(-len) %>% unlist %>% as.character())) %>% filter(abundance>0)
		# plot violins
		g1 <- ggplot(boot_X %>% filter(coexist==1), aes(y=abundance_noise, x= species, fill=species))+geom_violin(alpha=0.8)+
				geom_point(data=obs_X, aes(x=species, y = abundance), color="black",size=1,shape=5, alpha=0.6) +
				facet_grid(.~community, scales="free_x", space = "free_x")+scale_y_log10()+theme_bw()+ylab("Abundance")+
				theme(axis.title.x = element_blank(), legend.position = "none")
		# get median obs vs. pred values
		obs_pred <- obs_X_filt %>% group_by(community, species) %>% filter(abundance>0) %>% summarize(Observed = median(abundance)) %>%
					left_join(boot_X %>% mutate(community = as.character(community)) %>% filter(coexist==1, abundance>0) %>% group_by(community, species) %>% summarize(Predicted = median(abundance)), c("community","species"))
		g2 <- ggplot(obs_pred, aes(x=Observed, y=Predicted, fill=species)) + geom_point(shape=23, alpha=0.6, size=3) + geom_abline(intercept=0, slope=1)+theme_bw()+scale_y_log10()+scale_x_log10()
		# show the plots
		if("violin"%in%show_plot){
			print("Plotting violin plots")
			show(g1)
		}
		if("obs_pred"%in%show_plot){
			print("Plotting observed vs. predicted medians")
			show(g2)
		}
		if(return_plots){
			return(list(obs_pred = g2, violin = g1))
		}
	})
}
