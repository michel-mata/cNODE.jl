functions {
  // this function takes a B matrix and a row of E and predicted the endpoints
  vector predict_eq(matrix B1,       
               vector Ei,
               int N,
               vector y) {
  matrix[N,N] my_B;
  vector[N] predicted;
  real my_pred;
  // just copy over the Bs to avoid deep copy
  for(i in 1:N){
    for(j in 1:N){
      my_B[i,j] = B1[i,j];
    }
  }
  // if species i is present in Ei, keep B entries, otherwise zero out
  for(i in 1:N){
    if(Ei[i]==0){
      for(j in 1:N){
        my_B[i,j]=0;
        my_B[j,i]=0;
      }
      my_B[i,i]=1;
    }
  }
  // predict x by taking the negative row sum of the inverse
  predicted = inverse(my_B)*y;
  // if a species is incorrectly predicted to be absent, then
  // penalize heavily by assigning a tiny value
  for(i in 1:N){
    if(predicted[i]<=0){
      my_pred = predicted[i];
      predicted[i] = 10^(-20+my_pred);
    }
  }
  return predicted;
  }
}
data {
  int N;                      // number of species
  int M;                      // number of endpoints
  matrix[M,N] E;              // matrix of equilibrium    
  matrix[N,N] B_upper;        // matrix indicating entries with upper bound of zero
  matrix[N,N] B_lower;        // matrix indication entries with lower bound of zero
  vector[N] y;                // vector of -1s
  real maxB;                  // standard deviation for priors
}
parameters {
  matrix[N,N] B;                            // the B matrix
  vector<lower=0, upper=1.5>[N] sigmax;     // lognormal error, per species 
}
model {
  vector[N] my_x; // vector of predicted abundances
  // initialize sigma
  for(i in 1:N){
    sigmax[i] ~ normal(0,0.5)T[0,1.5]; 
  }
  // Set the priors based on upper/lower bounds
  for(i in 1:N){
    for(j in 1:N){
      if(B_upper[i,j]==0 && B_lower[i,j]!=0){
        B[i,j] ~ normal(0,maxB)T[,0];
      }
      else if(B_lower[i,j]==0 && B_upper[i,j]!=0){
        B[i,j] ~ normal(0,maxB)T[0,];
      }        
      else if(B_lower[i,j]==0 && B_upper[i,j]==0){
        B[i,j] ~ normal(0,1e-12);
      }        
      else{
        B[i,j] ~ normal(0,maxB);
      }
    }
  }
  for(i in 1:M){
    // predict the endpoint abundance for community i
    my_x = predict_eq(B, to_vector(E[i,]), N, y);
    for(j in 1:N){
      // only consider those present in endpoint i
      if(E[i,j]>0){
        // update the probability function by taking the log probability
        // of observing E[i,j] given my_x[j] and sigma[j]
        target +=  lognormal_lpdf(E[i,j] | log(my_x[j]), sigmax[j]); 
      }
    }
  }
}

