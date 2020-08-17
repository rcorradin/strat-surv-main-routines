#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

const double SQ6 = sqrt(6);
const double SQ3 = sqrt(3);
const double EMC = 0.57721566490153286060651209008; 

//----------------- UTILITY (PARTITIONS) ----------------------------------------------

//[[Rcpp::export]]
arma::mat psm(arma::mat M){
  // initialize results
  arma::mat result(M.n_cols, M.n_cols, arma::fill::zeros);
  
  for(arma::uword i = 0; i < M.n_cols; i++){
    for(arma::uword j = 0; j <= i; j++){
      result(i,j) = arma::accu(M.col(i) == M.col(j));
      result(j,i) = result(i,j);
    }
    Rcpp::checkUserInterrupt();
  }
  return(result / M.n_rows);
}

//[[Rcpp::export]]
arma::mat clean_partition(arma::mat M){
  
  arma::uvec index(M.n_cols);
  arma::vec tvec(M.n_cols);
  // initialize results
  arma::mat result(M.n_rows, M.n_cols, arma::fill::zeros);
  
  // for each row
  for(arma::uword k = 0; k < M.n_rows; k++){
    tvec = M.row(k).t();
    
    for(arma::uword j = 0; j < max(M.row(k)); j++){
      while((arma::accu(tvec == j + 1) == 0) && (arma::accu(tvec > j + 1) != 0)){
        index = find(tvec > j + 1);
        tvec(index) = tvec(index) - 1;
      }
    }
    
    result.row(k) = tvec.t();
    Rcpp::checkUserInterrupt();
  }
  return(result);
}

//[[Rcpp::export]]
arma::vec VI_LB(arma::mat C_mat, arma::mat psm_mat){
  
  arma::vec result(C_mat.n_rows);
  double f = 0.0;
  int n = psm_mat.n_cols;
  arma::vec tvec(n);
  
  for(arma::uword j = 0; j < C_mat.n_rows; j++){
    f = 0.0;
    for(arma::uword i = 0; i < n; i++){
      tvec = psm_mat.col(i);
      f += (log2(arma::accu(C_mat.row(j) == C_mat(j,i))) +
        log2(arma::accu(tvec)) -
        2 * log2(arma::accu(tvec.elem(arma::find(C_mat.row(j).t() == C_mat(j,i))))))/n;
    }
    result(j) = f;
    Rcpp::checkUserInterrupt();
  }
  return(result);
}

//----------------- SAMPLE INT (log scale) --------------------------------------------

// [[Rcpp::export]]
int rint_log(arma::vec lweights){
  
  for(arma::uword k = 0; k < lweights.n_elem; k++){
    if(!std::isfinite(lweights(k))){
      lweights(k) = - pow(10, 100);
    }
  }
  
  double u = arma::randu();
  arma::vec probs(lweights.n_elem);
  for(arma::uword k = 0; k < probs.n_elem; k++) {
    probs(k) = 1 / sum(exp(lweights - lweights(k)));
  }
  
  probs = arma::cumsum(probs);
  for(arma::uword k = 0; k < probs.n_elem; k++) {
    if(u <= probs[k]) {
      return k;
    }
  }
  return -1;
}

//----------------- PARA CLEAN --------------------------------------------------------

// [[Rcpp::export]]
void para_clean(arma::mat &param,
                arma::vec &clust) {
  int k = param.n_rows;
  int u_bound;
  
  // for all the used parameters
  for(arma::uword i = 0; i < k; i++){
    
    // if a cluster is empty
    if((int) arma::sum(clust == i) == 0){
      
      // find the last full cluster, then swap
      for(arma::uword j = k; j > i; j--){
        if((int) arma::sum(clust == j) != 0){
          
          // SWAPPING!!
          clust(arma::find(clust == j) ).fill(i);
          param.swap_rows(i,j);
          break;
        }
      }
    }
  }
  
  // reduce dimensions
  u_bound = 0;
  for(arma::uword i = 0; i < k; i++){
    if(arma::accu(clust == i) > 0){
      u_bound += 1;
    }
  }
  
  // resize object to the correct dimension
  param.resize(u_bound, param.n_cols);
}

//----------------- DENSITIES TYPE I --------------------------------------------------------

// [[Rcpp::export]]
double density_no_reg_typeI(double y,
                            double mu,
                            double zeta) {
  return ( (-exp(M_PI * (y - mu) / (zeta * SQ6)) + 
           M_PI * (y - mu) / (zeta * SQ6) - EMC ) + log(M_PI) - log(zeta * SQ6) );
}

// [[Rcpp::export]]
double density_reg_typeI(double y,
                         double mu,
                         double zeta, 
                         arma::vec theta,
                         arma::vec x){
  return ( (-exp(M_PI * (y - mu + arma::dot(theta, x)) / (zeta * SQ6)) + 
           M_PI * (y - mu + arma::dot(theta, x)) / (zeta * SQ6) - EMC ) + log(M_PI) - log(zeta * SQ6) );
}

// [[Rcpp::export]]
double surv_no_reg_typeI(double y,
                         double mu,
                         double zeta){
  return ((-exp( M_PI * ( y - mu ) / (SQ6 * zeta) - EMC )));
}

// [[Rcpp::export]]
double surv_reg_typeI(double y,
                      double mu,
                      double zeta,
                      arma::vec theta,
                      arma::vec x){
  return ((-exp( M_PI * ( y - mu + arma::dot(theta, x) ) / (SQ6 * zeta) - EMC )));
}

//----------------- DENSITIES LOG-LOGISTIC ---------------------------------------------------

// [[Rcpp::export]]
double density_no_reg_loglog(double y,
                             double mu,
                             double zeta) {
  return(log(M_PI / SQ3) - (M_PI * (y - mu) / (zeta * SQ3)) - 2 * log(1 + exp(-M_PI * (y - mu) / (zeta * SQ3))) - log(zeta));
}

// [[Rcpp::export]]
double density_reg_loglog(double y,
                          double mu,
                          double zeta,
                          arma::vec theta,
                          arma::vec x) {
  return(log(M_PI / SQ3) - (M_PI * (y - mu + arma::dot(theta, x)) / (zeta * SQ3)) - 
          2 * log(1 + exp(-M_PI * (y - mu + arma::dot(theta, x)) / (zeta * SQ3)))  - log(zeta));
}

// [[Rcpp::export]]
double surv_no_reg_loglog(double y,
                          double mu,
                          double zeta){
  return log(1 - 1 / (1 + exp(- M_PI * (y - mu) / (SQ3 * zeta))));
}

// [[Rcpp::export]]
double surv_reg_loglog(double y,
                       double mu,
                       double zeta,
                       arma::vec theta,
                       arma::vec x){
  return log(1 - 1 / (1 + exp(- M_PI * (y - mu + arma::dot(theta, x)) / (SQ3 * zeta))));
}

//----------------- DENSITIES NORMAL ---------------------------------------------------

// [[Rcpp::export]]
double density_no_reg_norm(double y,
                           double mu,
                           double zeta) {
  return R::dnorm((y - mu) / zeta, 0.0, 1.0, true) - log(zeta);
  // return log(arma::normpdf((y - mu) / zeta) / zeta);
}

// [[Rcpp::export]]
double density_reg_norm(double y,
                        double mu,
                        double zeta,
                        arma::vec theta,
                        arma::vec x) {
  return R::dnorm((y - mu + arma::dot(theta, x)) / zeta, 0.0, 1.0, true) - log(zeta);
  // return log(arma::normpdf((y - mu + arma::dot(theta, x)) / zeta) / zeta);
}

// [[Rcpp::export]]
double surv_no_reg_norm(double y,
                        double mu,
                        double zeta){
  return R::pnorm((y - mu) / zeta, 0.0, 1.0, 1, true);
  // return log(1 - arma::normcdf((y - mu) / zeta));
}

// [[Rcpp::export]]
double surv_reg_norm(double y,
                       double mu,
                       double zeta,
                       arma::vec theta,
                       arma::vec x){
  return R::pnorm((y - mu + arma::dot(theta, x)) / zeta, 0.0, 1.0, 1, true);
  // return log(1 - arma::normcdf((y - mu + arma::dot(theta, x)) / zeta));
}

//----------------- UPDATE CLUST ----------------------------------------------------------------

// [[Rcpp::export]]
void PY_clust_update_no_reg(int napprox,
                            double m0,
                            double s20,
                            double q0_zeta,
                            double q1_zeta,
                            double alpha,
                            double sigma,
                            arma::vec y,
                            arma::vec &clust,
                            arma::vec cens,
                            arma::mat &param,
                            int model){

  arma::mat temp_param;
  arma::vec probs;
  
  for(arma::uword i = 0; i < clust.n_elem; i++){
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    clust(i) = param.n_rows + 1;
    if(req_clean){
      para_clean(param, clust);
    }
    
    int k_temp = param.n_rows;
    
    // initialize common temporary locations
    // for Neal 8 algorithm
    arma::mat temp(napprox, param.n_cols);
    temp.col(0) = arma::randn(napprox) * sqrt(s20) + m0;
    temp.col(1) = 1.0 / arma::randg(napprox, arma::distr_param(q0_zeta, 1 / q1_zeta));
    
    // join the values 
    temp_param = arma::join_cols(param, temp);
    probs.resize(temp_param.n_rows);
    probs.fill(0.0);
    
    if(model == 1){
      
      // TYPE 1 MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
      }
    }else if(model == 2){
      
      // LOG-LOGISTIC MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
      }
    }else if(model == 3){
      
      // NORMAL MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
        }
      }
    }
    
    int temp_clust = rint_log(probs);
    clust(i) = (int) temp_clust;
    if(temp_clust >= k_temp){
      clust(i) = (int) k_temp;
      param = arma::join_cols(param, temp_param.row(temp_clust));
    }
  }
}

// [[Rcpp::export]]
void PY_clust_update_reg(int napprox,
                        double m0,
                        double s20,
                        arma::vec m_theta,
                        arma::vec s2_theta,
                        double q0_zeta,
                        double q1_zeta,
                        double alpha,
                        double sigma,
                        arma::vec y,
                        arma::vec &clust,
                        arma::mat covs,
                        arma::vec cens,
                        arma::mat &param,
                        int model){
  
  arma::mat temp_param;
  arma::vec probs;
  int cov_index = param.n_cols - 1;
  
  for(arma::uword i = 0; i < clust.n_elem; i++){
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    clust(i) = param.n_rows + 1;
    if(req_clean){
      para_clean(param, clust);
    }
    
    int k_temp = param.n_rows;
    
    // initialize common temporary locations
    // for Neal 8 algorithm
    arma::mat temp(napprox, param.n_cols);
    temp.col(0) = arma::randn(napprox) * sqrt(s20) + m0;
    temp.col(1) = 1.0 / arma::randg(napprox, arma::distr_param(q0_zeta, 1 / q1_zeta));
    for(arma::uword j = 0; j < param.n_cols - 2; j++){
      temp.col(j + 2) = arma::randn(napprox) * s2_theta(j) + m_theta(j);
    }
    
    // join the values 
    temp_param = arma::join_cols(param, temp);
    probs.resize(temp_param.n_rows);
    probs.fill(0.0);
    
    if(model == 1){
      
      // TYPE 1 MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_reg_typeI(y(i), temp_param(j,0), 
                temp_param(j,1), temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      }
    }else if(model == 2){
      
      // LOG LOGISTIC MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_reg_loglog(y(i), temp_param(j,0), 
                temp_param(j,1), temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      }
    }else if(model == 3){
      
      // NORMAL MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_reg_norm(y(i), temp_param(j,0), 
                temp_param(j,1), temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      }
    }
    
    int temp_clust = rint_log(probs);
    clust(i) = (int) temp_clust;
    if(temp_clust >= k_temp){
      clust(i) = (int) k_temp;
      param = arma::join_cols(param, temp_param.row(temp_clust));
    }
  }
}

// [[Rcpp::export]]
void PY_clust_update_common_reg(int napprox,
                                double m0,
                                double s20,
                                arma::vec m_theta,
                                arma::vec s2_theta,
                                double q0_zeta,
                                double q1_zeta,
                                double alpha,
                                double sigma,
                                arma::vec y,
                                arma::vec &clust,
                                arma::mat covs,
                                arma::vec cens,
                                arma::mat &param,
                                arma::vec common_reg,
                                int model){
  
  arma::mat temp_param;
  arma::vec probs;
  int cov_index = param.n_cols - 1;
  
  for(arma::uword i = 0; i < clust.n_elem; i++){
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    clust(i) = param.n_rows + 1;
    if(req_clean){
      para_clean(param, clust);
    }
    
    int k_temp = param.n_rows;
    
    // initialize common temporary locations
    // for Neal 8 algorithm
    arma::mat temp(napprox, 2);
    temp.col(0) = arma::randn(napprox) * sqrt(s20) + m0;
    temp.col(1) = 1.0 / arma::randg(napprox, arma::distr_param(q0_zeta, 1 / q1_zeta));
    for(arma::uword j = 0; j < param.n_cols - 2; j++){
      temp.col(j + 2) = arma::randn(napprox) * s2_theta(j) + m_theta(j);
    }
    
    // join the values 
    temp_param = arma::join_cols(param, temp);
    probs.resize(temp_param.n_rows);
    probs.fill(0.0);
    
    if(model == 1){
      
      // TYPE 1 MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_reg_typeI(y(i), temp_param(j,0), 
                temp_param(j,1), common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      }
    }else if(model == 2){
      
      // LOG LOGISTIC MODEL
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_reg_loglog(y(i), temp_param(j,0), 
                temp_param(j,1), common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      }
    }else if(model == 3){
      
      if(cens(i) == 1){
        // compute the probabilities 
        // not censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + density_reg_norm(y(i), temp_param(j,0), 
                temp_param(j,1), common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + density_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      } else {
        // compute the probabilities 
        // censored
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - sigma) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha + k_temp * sigma) - log(napprox) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      }
    }
    
    int temp_clust = rint_log(probs);
    clust(i) = (int) temp_clust;
    if(temp_clust >= k_temp){
      clust(i) = (int) k_temp;
      param = arma::join_cols(param, temp_param.row(temp_clust));
    }
  }
}

//------------------------ ACCELERATE ----------------------------------------------------------
// MH-step to reshuffle the parameters for each cluster
// Normal proposal
// log trasf for zeta

void accelerate_no_reg(arma::vec y,
                       arma::mat &param,
                       arma::vec clust,
                       arma::vec cens,
                       arma::vec prior_mean,
                       arma::mat prior_var,
                       arma::vec accelerate_var,
                       arma::vec &accelerate_count,
                       arma::uword iter,
                       int model){
  
  arma::vec tvec, tnew;
  double tprob, utemp;
  arma::mat rooti  = arma::trans(arma::inv(trimatu(arma::chol(prior_var))));
  
  for(arma::uword j = 0; j < param.n_rows; j++){
    tvec = param.row(j).t();
    tvec(1) = log(tvec(1));
    
    arma::vec cdata  = rooti * (tvec - prior_mean) ;
    double tprob     = - (tvec.n_elem / 2.0) * log(2.0 * M_PI) - 0.5 * arma::dot(cdata, cdata) + arma::sum(log(rooti.diag()));
    
    // MH step with normal proposal (log of zeta)
    tnew = arma::randn(2) % sqrt(accelerate_var) + tvec;
    
    if(model == 1){
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_no_reg_typeI(y(i), tnew(0), exp(tnew(1)));
            tprob -= density_no_reg_typeI(y(i), param(j,0), param(j,1));
          } else {
            tprob += surv_no_reg_typeI(y(i), tnew(0), exp(tnew(1)));
            tprob -= surv_no_reg_typeI(y(i), param(j,0), param(j,1));
          }
        }
      }
    } else if(model == 2) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_no_reg_loglog(y(i), tnew(0), exp(tnew(1)));
            tprob -= density_no_reg_loglog(y(i), param(j,0), param(j,1));
          } else {
            tprob += surv_no_reg_loglog(y(i), tnew(0), exp(tnew(1)));
            tprob -= surv_no_reg_loglog(y(i), param(j,0), param(j,1));
          }
        }
      }
    } else if(model == 3) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_no_reg_norm(y(i), tnew(0), exp(tnew(1)));
            tprob -= density_no_reg_norm(y(i), param(j,0), param(j,1));
          } else {
            tprob += surv_no_reg_norm(y(i), tnew(0), exp(tnew(1)));
            tprob -= surv_no_reg_norm(y(i), param(j,0), param(j,1));
          }
        }
      }
    }
    
    if(log(arma::randu()) < std::min(0.0, tprob)){
      param(j,0) = tnew(0);
      param(j,1) = exp(tnew(1));
      accelerate_count(iter) += 1;
    }
  }
}

// REGRESSION MODEL - GROUP SPECIFIC COEFFICIENTS
void accelerate_reg(arma::vec y,
                    arma::mat covs,
                    arma::mat &param,
                    arma::vec clust,
                    arma::vec cens,
                    arma::vec prior_mean,
                    arma::mat prior_var,
                    arma::vec accelerate_var,
                    arma::vec &accelerate_count,
                    arma::uword iter,
                    int model){
  
  arma::vec tvec, tnew;
  double tprob, utemp;
  int cov_index = param.n_cols - 1;
  arma::mat rooti  = arma::trans(arma::inv(trimatu(arma::chol(prior_var))));
  
  for(arma::uword j = 0; j < param.n_rows; j++){
    tvec = param.row(j).t();
    tvec(1) = log(tvec(1));
    
    arma::vec cdata  = rooti * (tvec - prior_mean) ;
    double tprob     = - (tvec.n_elem / 2.0) * log(2.0 * M_PI) - 0.5 * arma::dot(cdata, cdata) + arma::sum(log(rooti.diag()));
    
    // MH step with normal proposal (log of zeta)
    tnew = arma::randn(tvec.n_elem) % sqrt(accelerate_var) + tvec;
    
    if(model == 1){
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_typeI(y(i), tnew(0), exp(tnew(1)), tnew(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= density_reg_typeI(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          } else {
            tprob += surv_reg_typeI(y(i), tnew(0), exp(tnew(1)), tnew(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= surv_reg_typeI(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          }
        }
      }
    } else if(model == 2) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_loglog(y(i), tnew(0), exp(tnew(1)), tnew(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= density_reg_loglog(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          } else {
            tprob += surv_reg_loglog(y(i), tnew(0), exp(tnew(1)), tnew(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= surv_reg_loglog(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          }
        }
      }
    } else if(model == 3) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_norm(y(i), tnew(0), exp(tnew(1)), tnew(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= density_reg_norm(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          } else {
            tprob += surv_reg_norm(y(i), tnew(0), exp(tnew(1)), tnew(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= surv_reg_norm(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          }
        }
      }
    }
    
    if(log(arma::randu()) < std::min(0.0, tprob)){
      param.row(j) = tnew.t();
      param(j,1) = exp(tnew(1));
      accelerate_count(iter) += 1;
    }
  }
}

// REGRESSION MODEL - COMMON COEFFICIENTS
//[[Rcpp::export]]
void accelerate_common_reg(arma::vec y,
                           arma::mat covs,
                           arma::mat &param,
                           arma::vec &common_reg,
                           arma::vec clust,
                           arma::vec cens,
                           arma::vec prior_mean,
                           arma::mat prior_var,
                           arma::vec accelerate_var,
                           arma::vec &accelerate_count,
                           arma::uword iter,
                           int model){
  
  arma::vec tvec, tnew;
  double tprob, utemp;
  arma::mat rooti  = arma::trans(arma::inv(trimatu(arma::chol(prior_var.submat(0,0,1,1)))));
  
  for(arma::uword j = 0; j < param.n_rows; j++){
    tvec = param.row(j).t();
    tvec(1) = log(tvec(1));
   
    arma::vec cdata  = rooti * (tvec - prior_mean.subvec(0,1)) ;
    double tprob     = - (tvec.n_elem / 2.0) * log(2.0 * M_PI) - 0.5 * arma::dot(cdata, cdata) + arma::sum(log(rooti.diag()));
    
    // MH step with normal proposal (log of zeta)
    tnew = arma::randn(2) % sqrt(accelerate_var(arma::span(0, 1))) + tvec;
    
    if(model == 1){
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_typeI(y(i), tnew(0), exp(tnew(1)), common_reg, covs.row(i).t());
            tprob -= density_reg_typeI(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          } else {
            tprob += surv_reg_typeI(y(i), tnew(0), exp(tnew(1)), common_reg, covs.row(i).t());
            tprob -= surv_reg_typeI(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          }
        }
      }
    } else if(model == 2) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_loglog(y(i), tnew(0), exp(tnew(1)), common_reg, covs.row(i).t());
            tprob -= density_reg_loglog(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          } else {
            tprob += surv_reg_loglog(y(i), tnew(0), exp(tnew(1)), common_reg, covs.row(i).t());
            tprob -= surv_reg_loglog(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          }
        }
      }
    } else if(model == 3) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_norm(y(i), tnew(0), exp(tnew(1)), common_reg, covs.row(i).t());
            tprob -= density_reg_norm(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          } else {
            tprob += surv_reg_norm(y(i), tnew(0), exp(tnew(1)), common_reg, covs.row(i).t());
            tprob -= surv_reg_norm(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          }
        }
      }
    }
    if(log(arma::randu()) < std::min(0.0, tprob)){
      param(j,0) = tnew(0);
      param(j,1) = exp(tnew(1));
      accelerate_count(iter) += 1;
    }
  }
  
  arma::vec regnew = arma::randn(common_reg.n_elem) % accelerate_var.subvec(2, accelerate_var.n_elem - 1) + common_reg;
  tprob = 0.0;
  if(model == 1){
    for(arma::uword i = 0; i < clust.n_elem; i++){
      if(cens(i) == 1){
        tprob += density_reg_typeI(y(i), param(clust(i),0), param(clust(i),1), regnew, covs.row(i).t());
        tprob -= density_reg_typeI(y(i), param(clust(i),0), param(clust(i),1), common_reg, covs.row(i).t());
      } else {
        tprob += surv_reg_typeI(y(i), param(clust(i),0), param(clust(i),1), regnew, covs.row(i).t());
        tprob -= surv_reg_typeI(y(i), param(clust(i),0), param(clust(i),1), common_reg, covs.row(i).t());
      }
    }
  } else if(model == 2) {
    for(arma::uword i = 0; i < clust.n_elem; i++){
      if(cens(i) == 1){
        tprob += density_reg_loglog(y(i), param(clust(i),0), param(clust(i),1), regnew, covs.row(i).t());
        tprob -= density_reg_loglog(y(i), param(clust(i),0), param(clust(i),1), common_reg, covs.row(i).t());
      } else {
        tprob += surv_reg_loglog(y(i), param(clust(i),0), param(clust(i),1), regnew, covs.row(i).t());
        tprob -= surv_reg_loglog(y(i), param(clust(i),0), param(clust(i),1), common_reg, covs.row(i).t());
      }
    }
  } else if(model == 3) {
    for(arma::uword i = 0; i < clust.n_elem; i++){
      if(cens(i) == 1){
        tprob += density_reg_norm(y(i), param(clust(i),0), param(clust(i),1), regnew, covs.row(i).t());
        tprob -= density_reg_norm(y(i), param(clust(i),0), param(clust(i),1), common_reg, covs.row(i).t());
      } else {
        tprob += surv_reg_norm(y(i), param(clust(i),0), param(clust(i),1), regnew, covs.row(i).t());
        tprob -= surv_reg_norm(y(i), param(clust(i),0), param(clust(i),1), common_reg, covs.row(i).t());
      }
    }
  }
  if(log(arma::randu()) < std::min(0.0, tprob)){
    common_reg = regnew;
  }
}

//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
//------------------------------------ MAINS ---------------------------------------------------
//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List PY_main_no_reg(arma::vec y,
                       arma::vec cens,
                       int niter, 
                       int nburn,
                       int napprox,
                       double m0, 
                       double s20,
                       double q0_zeta, 
                       double q1_zeta, 
                       double alpha,
                       double sigma,
                       arma::vec accelerate_var,
                       arma::vec prior_mean,
                       arma::mat prior_var,
                       int model){

  // initialize the results
  arma::mat clust_result(niter - nburn, y.n_elem);
  arma::vec accelerate_count(niter, arma::fill::zeros);
  
  // initialize the quantities
  arma::mat param;
  arma::vec clust(y.n_elem);
  
  clust.fill(0);
  param.resize(1,2);
  param(0,0) = arma::randn() * s20 + m0;
  param(0,1) = 1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0);
  
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  // main loop
  for(arma::uword iter = 0; iter < niter; iter++){
    
    // update clusters
    PY_clust_update_no_reg(napprox, m0, s20, q0_zeta, q1_zeta, alpha, sigma, y, clust, cens, param, model);
    
    // accelerate
    accelerate_no_reg(y, param, clust, cens, prior_mean, prior_var, accelerate_var, accelerate_count, iter, model);
      
    if(iter >= nburn){
      clust_result.row(iter - nburn) = clust.t();
    }
    
    // print status and check interrupt
    if((iter + 1) % nupd == 0){
      current_s = clock();
      Rcpp::Rcout << "Completed:\t" << (iter + 1) << "/" << niter << " - in " <<
        double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
    }
    Rcpp::checkUserInterrupt();
  }
  
  // return the results
  
  Rcpp::List results;
  results["clusts"] = clust_result;
  results["accelerate_count"] = accelerate_count;
  results["time"] = double(current_s-start_s)/CLOCKS_PER_SEC;
  return results;
}

// [[Rcpp::export]]
Rcpp::List PY_main_reg(arma::vec y, 
                      arma::mat covs,
                      arma::vec cens,
                      int niter, 
                      int nburn,
                      int napprox,
                      double m0, 
                      double s20, 
                      arma::vec m_theta,
                      arma::vec s2_theta,
                      double q0_zeta, 
                      double q1_zeta, 
                      double alpha,
                      double sigma,
                      arma::vec accelerate_var,
                      arma::vec prior_mean,
                      arma::mat prior_var,
                      int model){
  
  // initialize the results
  arma::mat clust_result(niter - nburn, y.n_elem);
  arma::vec accelerate_count(niter, arma::fill::zeros);
  
  // initialize the quantities
  arma::mat param;
  arma::vec clust(y.n_elem);
  
  clust.fill(0);
  param.resize(1,2 + covs.n_cols);
  param(0,0) = arma::randn() * s20 + m0;
  param(0,1) = 1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0);
  param(0, arma::span(2, param.n_cols - 1)) = arma::randn(param.n_cols - 2).t() ;
  
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  // main loop
  for(arma::uword iter = 0; iter < niter; iter++){

    // update clusters
    PY_clust_update_reg(napprox, m0, s20, m_theta, s2_theta, q0_zeta, q1_zeta, 
                           alpha, sigma, y, clust, covs, cens, param, model);
    
    // accelerate
    accelerate_reg(y, covs, param, clust, cens, prior_mean, prior_var, accelerate_var, accelerate_count, iter, model);
    
    if(iter >= nburn){
      clust_result.row(iter - nburn) = clust.t();
    }
    
    // print status and check interrupt
    if((iter + 1) % nupd == 0){
      current_s = clock();
      Rcpp::Rcout << "Completed:\t" << (iter + 1) << "/" << niter << " - in " <<
        double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
    }
    Rcpp::checkUserInterrupt();
  }
  
  // return the results
  
  Rcpp::List results;
  results["clusts"] = clust_result;
  results["accelerate_count"] = accelerate_count;
  results["time"] = double(current_s-start_s)/CLOCKS_PER_SEC;
  return results;
}

// [[Rcpp::export]]
Rcpp::List main_common_reg(arma::vec y, 
                           arma::mat covs,
                           arma::vec cens,
                           int niter, 
                           int nburn,
                           int napprox,
                           double m0, 
                           double s20, 
                           arma::vec m_theta,
                           arma::vec s2_theta,
                           double q0_zeta, 
                           double q1_zeta, 
                           double alpha,
                           double sigma,
                           arma::vec accelerate_var,
                           arma::vec prior_mean,
                           arma::mat prior_var,
                           int model){
  
  // initialize the results
  arma::mat clust_result(niter - nburn, y.n_elem);
  arma::vec accelerate_count(niter, arma::fill::zeros);
  
  // initialize the quantities
  arma::mat param;
  arma::vec clust(y.n_elem);
  arma::vec common_reg(covs.n_cols);
  
  clust.fill(0);
  param.resize(1,2);
  param(0,0) = arma::randn() * s20 + m0;
  param(0,1) = 1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0);
  common_reg.fill(1.0);
  
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  // main loop
  for(arma::uword iter = 0; iter < niter; iter++){

    // update clusters
    PY_clust_update_common_reg(napprox, m0, s20, m_theta, s2_theta, q0_zeta, q1_zeta,
                     alpha, sigma, y, clust, covs, cens, param, common_reg, model);
    
    // accelerate
    accelerate_common_reg(y, covs, param, common_reg, clust, cens, prior_mean, prior_var, accelerate_var, accelerate_count, iter, model);
    
    if(iter >= nburn){
      clust_result.row(iter - nburn) = clust.t();
    }
    
    // print status and check interrupt
    if((iter + 1) % nupd == 0){
      current_s = clock();
      Rcpp::Rcout << "Completed:\t" << (iter + 1) << "/" << niter << " - in " <<
        double(current_s-start_s)/CLOCKS_PER_SEC << " sec\n";
    }
    Rcpp::checkUserInterrupt();
  }
  
  // return the results
  
  Rcpp::List results;
  results["clusts"] = clust_result;
  results["accelerate_count"] = accelerate_count;
  results["time"] = double(current_s-start_s)/CLOCKS_PER_SEC;
  return results;
}

