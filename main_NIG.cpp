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

int rint_log(arma::vec lweights){
  
  double u = arma::randu();
  double maxLP = max(lweights);
  arma::vec lprobs = lweights - maxLP;
  arma::vec probs = exp(lprobs);
  
  // Rcpp::Rcout << "\n\n" << probs.t();
  probs /= sum(probs);
  probs = cumsum(probs);
  
  // Rcpp::Rcout << "\n\n" << probs.t();
  
  double p_sum = 0.0;
  for(arma::uword k = 0; k < probs.n_elem; k++) {
    if(u <= probs(k)) {
      return k;
    }
  }
  
  return -1;
}

//----------------- PARA CLEAN --------------------------------------------------------

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
double density_no_reg_typeI(double y,
                            double mu,
                            double zeta) {
  return(
    -exp( (y - mu) * M_PI / (zeta * SQ6) - EMC ) + 
      ((y - mu) * M_PI / (zeta * SQ6) - EMC)  + log(M_PI) - log(zeta * SQ6)
  );
}

double density_reg_typeI(double y,
                         double mu,
                         double zeta, 
                         arma::vec theta,
                         arma::vec x){
  return(
    -exp( (y - mu + arma::dot(theta, x)) * M_PI / (zeta * SQ6) - EMC ) + 
      ((y - mu + arma::dot(theta, x)) * M_PI / (zeta * SQ6) - EMC)  + log(M_PI) - log(zeta * SQ6)
  );
}

double surv_no_reg_typeI(double y,
                         double mu,
                         double zeta){
  return(
    -exp( (y - mu) * M_PI / (zeta * SQ6) - EMC )
  );
}

double surv_reg_typeI(double y,
                      double mu,
                      double zeta,
                      arma::vec theta,
                      arma::vec x){
  return(
    -exp( (y - mu + arma::dot(theta, x)) * M_PI / (zeta * SQ6) - EMC )
  );
}

//----------------- DENSITIES LOG-LOGISTIC ---------------------------------------------------

double density_no_reg_loglog(double y,
                             double mu,
                             double zeta) {
  return(log(M_PI / SQ3) - (M_PI * (y - mu) / (zeta * SQ3)) - 2 * log(1 + exp(-M_PI * (y - mu) / (zeta * SQ3))) - log(zeta));
}

double density_reg_loglog(double y,
                          double mu,
                          double zeta,
                          arma::vec theta,
                          arma::vec x) {
  return(log(M_PI / SQ3) - (M_PI * (y - mu + arma::dot(theta, x)) / (zeta * SQ3)) - 
         2 * log(1 + exp(-M_PI * (y - mu + arma::dot(theta, x)) / (zeta * SQ3)))  - log(zeta));
}

double surv_no_reg_loglog(double y,
                          double mu,
                          double zeta){
  return log(1 - 1 / (1 + exp(- M_PI * (y - mu) / (SQ3 * zeta))));
}

double surv_reg_loglog(double y,
                       double mu,
                       double zeta,
                       arma::vec theta,
                       arma::vec x){
  return log(1 - 1 / (1 + exp(- M_PI * (y - mu + arma::dot(theta, x)) / (SQ3 * zeta))));
}

//----------------- DENSITIES NORMAL ---------------------------------------------------

double density_no_reg_norm(double y,
                           double mu,
                           double zeta) {
  return R::dnorm((y - mu) / zeta, 0.0, 1.0, true) - log(zeta);
}

double density_reg_norm(double y,
                        double mu,
                        double zeta,
                        arma::vec theta,
                        arma::vec x) {
  return R::dnorm((y - mu + arma::dot(theta, x)) / zeta, 0.0, 1.0, true) - log(zeta);
}

double surv_no_reg_norm(double y,
                        double mu,
                        double zeta){
  return R::pnorm((y - mu) / zeta, 0.0, 1.0, 1, true);
}

double surv_reg_norm(double y,
                     double mu,
                     double zeta,
                     arma::vec theta,
                     arma::vec x){
  return R::pnorm((y - mu + arma::dot(theta, x)) / zeta, 0.0, 1.0, 1, true);
}

//----------------- UPDATE U --------------------------------------------------------

void u_update(double &u,
              double s2v,
              double alpha,
              double tau,
              arma::vec clust,
              int &acc_rate_u){
  
  double v_old = log(u);
  double v_temp = arma::randn() * sqrt(s2v) + v_old;
  double u_temp = exp(v_temp);
  arma::vec t_unique = arma::unique(clust);
  double k = t_unique.n_elem;
  double n = clust.n_elem;
  
  double acc_ratio_log = std::min(0.0, 
                                  n * (v_temp - v_old) - alpha * (sqrt(exp(v_temp) + tau) - sqrt(exp(v_old) + tau)) +
                                    (k / 2 - n) * (log(exp(v_temp) + tau) - log(exp(v_old) + tau))
  );
  if(log(arma::randu()) <= acc_ratio_log){
    u = u_temp;
    acc_rate_u += 1;
  }
}

//----------------- UPDATE TAU --------------------------------------------------------

void tau_update(double &tau,
                double q0_tau,
                double q1_tau,
                double alpha,
                double u,
                double s2t,
                arma::vec clust,
                int &acc_rate_tau){
  
  double t_old = log(tau);
  double t_temp = arma::randn() * sqrt(s2t) + t_old;
  double tau_temp = exp(t_temp);
  arma::vec t_unique = arma::unique(clust);
  double k = t_unique.n_elem;
  double n = clust.n_elem;
  
  double acc_ratio_log = std::min(0.0, 
                                  - alpha * (sqrt(u + tau_temp) - sqrt(tau_temp) - sqrt(u + tau) + sqrt(tau)) +
                                    (k / 2 - n) * (log(u + tau_temp) - log(u + tau)) - 
                                    q1_tau * (tau_temp - tau) + q0_tau * (t_temp - t_old)
  );
  if(log(arma::randu()) <= acc_ratio_log){
    tau = exp(t_temp);
    acc_rate_tau += 1;
  }
}

//----------------- UPDATE CLUST ----------------------------------------------------------------

void clust_update_no_reg(int napprox,
                         double m0,
                         double s20,
                         double q0_zeta,
                         double q1_zeta,
                         double u,
                         double tau,
                         double alpha,
                         arma::vec y,
                         arma::vec &clust,
                         arma::vec cens,
                         arma::mat &param,
                         int model){
  
  arma::mat temp_param;
  arma::mat temp(napprox,2);
  arma::vec probs;
  
  for(arma::uword i = 0; i < clust.n_elem; i++){
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    temp.col(0) = arma::randn(napprox) * sqrt(s20) + m0;
    for(arma::uword j = 0; j < napprox; j++){
      temp(j,1) = sqrt(1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0) );
    }
    
    if(req_clean){
      temp.row(0) = param.row(clust(i));
      clust(i) = param.n_rows + 1;
      para_clean(param, clust);
    }
    
    int k_temp = param.n_rows;
    
    // join the values 
    temp_param = arma::join_cols(param, temp);
    probs.resize(temp_param.n_rows);
    probs.fill(0.0);
    
    if(model == 1){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_no_reg_typeI(y(i), temp_param(j,0), temp_param(j,1));
        }
      }
    }else if(model == 2){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_no_reg_loglog(y(i), temp_param(j,0), temp_param(j,1));
        }
      }
    }else if(model == 3){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_no_reg_norm(y(i), temp_param(j,0), temp_param(j,1));
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

void clust_update_common_reg(int napprox,
                             double m0,
                             double s20,
                             arma::vec m_theta,
                             arma::vec s2_theta,
                             double q0_zeta,
                             double q1_zeta,
                             double u,
                             double tau,
                             double alpha,
                             arma::vec y,
                             arma::vec &clust,
                             arma::mat covs,
                             arma::vec cens,
                             arma::mat &param,
                             arma::vec common_reg,
                             int model){
  
  arma::mat temp_param;
  arma::mat temp(napprox, 2);
  arma::vec probs;
  int cov_index = param.n_cols - 1;
  
  for(arma::uword i = 0; i < clust.n_elem; i++){
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    temp.col(0) = arma::randn(napprox) * sqrt(s20) + m0;
    for(arma::uword j = 0; j < napprox; j++){
      temp(j,1) = sqrt(1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0) );
    }
    
    if(req_clean){
      temp.row(0) = param.row(clust(i));
      clust(i) = param.n_rows + 1;
      para_clean(param, clust);
    }
    
    int k_temp = param.n_rows;
    
    // join the values 
    temp_param = arma::join_cols(param, temp);
    probs.resize(temp_param.n_rows);
    probs.fill(0.0);
    
    // compute probs
    if(model == 1){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_reg_typeI(y(i), temp_param(j,0), 
                temp_param(j,1), common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      }
    }else if(model == 2){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_reg_loglog(y(i), temp_param(j,0), 
                temp_param(j,1), common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      }
    }else if(model == 3){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_reg_norm(y(i), temp_param(j,0), 
                temp_param(j,1), common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                common_reg, covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
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

// Specific coefficients
// clust update function
void clust_update_reg(int napprox,
                      double m0,
                      double s20,
                      arma::vec m_theta,
                      arma::vec s2_theta,
                      double q0_zeta,
                      double q1_zeta,
                      double u,
                      double tau,
                      double alpha,
                      arma::vec y,
                      arma::vec &clust,
                      arma::mat covs,
                      arma::vec cens,
                      arma::mat &param,
                      int model){
  
  arma::mat temp_param;
  arma::vec probs;
  int cov_index = param.n_cols - 1;
  arma::mat temp(napprox, param.n_cols);
  
  for(arma::uword i = 0; i < clust.n_elem; i++){
    
    bool req_clean = false;
    if(arma::sum(clust == clust[i]) == 1){
      req_clean = true;
    }
    
    temp.col(0) = arma::randn(napprox) * sqrt(s20) + m0;
    for(arma::uword j = 0; j < napprox; j++){
      temp(j,1) = sqrt(1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0) );
    }
    for(arma::uword j = 0; j < param.n_cols - 2; j++){
      temp.col(j + 2) = arma::randn(napprox) * sqrt(s2_theta(j)) + m_theta(j);
    }
    
    if(req_clean){
      temp.row(0) = param.row(clust(i));
      clust(i) = param.n_rows + 1;
      para_clean(param, clust);
    }
    
    int k_temp = param.n_rows;
    
    // join the values 
    temp_param = arma::join_cols(param, temp);
    probs.resize(temp_param.n_rows);
    probs.fill(0.0);
    
    if(model == 1){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_reg_typeI(y(i), temp_param(j,0), 
                temp_param(j,1), temp_param(j, arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_reg_typeI(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      }
    } else if(model == 2){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_reg_loglog(y(i), temp_param(j,0), 
                temp_param(j,1), temp_param(j, arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_reg_loglog(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      }
    } else if(model == 3){
      if(cens(i) == 1){
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + density_reg_norm(y(i), temp_param(j,0), 
                temp_param(j,1), temp_param(j, arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + density_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
      } else {
        for(arma::uword j = 0; j < param.n_rows; j++){
          probs(j) = log(arma::accu(clust == j) - 0.5) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
                temp_param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
        }
        for(arma::uword j = param.n_rows; j < temp_param.n_rows; j++){
          probs(j) = log(alpha * sqrt(u + tau)) - log(2 * napprox) + surv_reg_norm(y(i), temp_param(j,0), temp_param(j,1), 
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

//------------------------ ACCELERATE ----------------------------------------------------------
// MH-step to reshuffle the parameters for each cluster
// Normal proposal
// log trasf for zeta

void accelerate_no_reg(arma::vec y,
                       arma::mat &param,
                       arma::vec clust,
                       arma::vec cens,
                       double m0, 
                       double s20, 
                       double q0_zeta, 
                       double q1_zeta,
                       arma::vec accelerate_var,
                       arma::vec &accelerate_count,
                       arma::uword iter,
                       int model){
  
  double tprob;
  double temp_zeta, new_zeta, temp_mu, new_mu;
  arma::vec temp_param(param.n_cols);
  
  for(arma::uword j = 0; j < param.n_rows; j++){
    
    // ----------- MU -----------------------
    temp_mu = param(j,0);
    new_mu  = arma::randn() * sqrt(accelerate_var(0)) + temp_mu;
    tprob = log(arma::normpdf(new_mu, m0, sqrt(s20))) - log(arma::normpdf(temp_mu, m0, sqrt(s20)));
    temp_param(0) = new_mu;
    
    // ----------- ZETA ---------------------
    temp_zeta = log(pow(param(j,1), 2));
    new_zeta  = arma::randn() * sqrt(accelerate_var(1)) + temp_zeta;
    tprob += - (q0_zeta) * new_zeta - q1_zeta / exp(new_zeta) + (q0_zeta) * temp_zeta + q1_zeta / exp(temp_zeta);
    new_zeta = sqrt(exp(new_zeta));
    temp_param(1) = new_zeta;
    
    // ------------ LIKELIHOOD --------------
    if(model == 1){
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_no_reg_typeI(y(i), temp_param(0), temp_param(1));
            tprob -= density_no_reg_typeI(y(i), param(j,0), param(j,1));
          } else {
            tprob += surv_no_reg_typeI(y(i), temp_param(0), temp_param(1));
            tprob -= surv_no_reg_typeI(y(i), param(j,0), param(j,1));
          }
        }
      }
    } else if(model == 2) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_no_reg_loglog(y(i), temp_param(0), temp_param(1));
            tprob -= density_no_reg_loglog(y(i), param(j,0), param(j,1));
          } else {
            tprob += surv_no_reg_loglog(y(i), temp_param(0), temp_param(1));
            tprob -= surv_no_reg_loglog(y(i), param(j,0), param(j,1));
          }
        }
      }
    } else if(model == 3) {
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_no_reg_norm(y(i), temp_param(0), temp_param(1));
            tprob -= density_no_reg_norm(y(i), param(j,0), param(j,1));
          } else {
            tprob += surv_no_reg_norm(y(i), temp_param(0), temp_param(1));
            tprob -= surv_no_reg_norm(y(i), param(j,0), param(j,1));
          }
        }
      }
    }
    
    if(log(arma::randu()) < std::min(0.0, tprob)){
      param.row(j) = temp_param.t();
    }
  }
}

// REGRESSION MODEL - COMMON COEFFICIENTS
void accelerate_common_reg(arma::vec y,
                           arma::mat covs,
                           arma::mat &param,
                           arma::vec &common_reg,
                           arma::vec clust,
                           arma::vec cens,
                           double m0, 
                           double s20, 
                           arma::vec m_theta,
                           arma::vec s2_theta,
                           double q0_zeta, 
                           double q1_zeta,
                           arma::vec accelerate_var,
                           arma::vec &accelerate_count,
                           arma::uword iter,
                           int model){
  
  double tprob;
  double temp_zeta, new_zeta, temp_mu, new_mu;
  arma::vec temp_param(param.n_cols);
  
  for(arma::uword j = 0; j < param.n_rows; j++){
    
    // ----------- MU -----------------------
    temp_mu = param(j,0);
    new_mu  = arma::randn() * sqrt(accelerate_var(0)) + temp_mu;
    tprob = log(arma::normpdf(new_mu, m0, sqrt(s20))) - log(arma::normpdf(temp_mu, m0, sqrt(s20)));
    temp_param(0) = new_mu;
    
    // ----------- ZETA ---------------------
    temp_zeta = log(pow(param(j,1), 2));
    new_zeta  = arma::randn() * sqrt(accelerate_var(1)) + temp_zeta;
    tprob += - (q0_zeta) * new_zeta - q1_zeta / exp(new_zeta) + (q0_zeta) * temp_zeta + q1_zeta / exp(temp_zeta);
    new_zeta = sqrt(exp(new_zeta));
    temp_param(1) = new_zeta;
    
    // ----------- LIKELIHOOD ---------------
    for(arma::uword i = 0; i < clust.n_elem; i++){
      if(model == 1){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_typeI(y(i), temp_param(0), temp_param(1), common_reg, covs.row(i).t());
            tprob -= density_reg_typeI(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          } else {
            tprob += surv_reg_typeI(y(i), temp_param(0), temp_param(1), common_reg, covs.row(i).t());
            tprob -= surv_reg_typeI(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          }
        }
      } else if(model == 2){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_loglog(y(i), temp_param(0), temp_param(1), common_reg, covs.row(i).t());
            tprob -= density_reg_loglog(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          } else {
            tprob += surv_reg_loglog(y(i), temp_param(0), temp_param(1), common_reg, covs.row(i).t());
            tprob -= surv_reg_loglog(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          }
        }
      } else if(model == 3){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_norm(y(i), temp_param(0), temp_param(1), common_reg, covs.row(i).t());
            tprob -= density_reg_norm(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          } else {
            tprob += surv_reg_norm(y(i), temp_param(0), temp_param(1), common_reg, covs.row(i).t());
            tprob -= surv_reg_norm(y(i), param(j,0), param(j,1), common_reg, covs.row(i).t());
          }
        }
      }
    }
    
    if(log(arma::randu()) < std::min(0.0, tprob)){
      param.row(j) = temp_param.t();
    }
  }
  
  // ----------- BETA ---------------------
  arma::vec regnew(common_reg.n_elem);
  for(arma::uword l = 0; l < common_reg.n_elem; l++){
    regnew(l) = arma::randn() * sqrt(accelerate_var(l + 2)) + common_reg(l);
    tprob = log(arma::normpdf(regnew(l), m_theta(l), sqrt(s2_theta(l)))) -
      log(arma::normpdf(common_reg(l), m_theta(l), sqrt(s2_theta(l))));
  }
  
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

// REGRESSION MODEL - GROUP SPECIFIC COEFFICIENTS
void accelerate_reg(arma::vec y,
                    arma::mat covs,
                    arma::mat &param,
                    arma::vec clust,
                    arma::vec cens,
                    double m0, 
                    double s20, 
                    arma::vec m_theta,
                    arma::vec s2_theta,
                    double q0_zeta, 
                    double q1_zeta,
                    arma::vec accelerate_var,
                    arma::vec &accelerate_count,
                    arma::uword iter,
                    int model){
  
  double temp_coef, new_coef;
  double tprob, utemp;
  int cov_index = param.n_cols - 1;
  double temp_zeta, new_zeta, temp_mu, new_mu;
  arma::vec temp_param(param.n_cols);
  
  for(arma::uword j = 0; j < param.n_rows; j++){
    
    // ----------- MU -----------------------
    temp_mu = param(j,0);
    new_mu  = arma::randn() * sqrt(accelerate_var(0)) + temp_mu;
    tprob = log(arma::normpdf(new_mu, m0, sqrt(s20))) - log(arma::normpdf(temp_mu, m0, sqrt(s20)));
    temp_param(0) = new_mu;
    
    // ----------- ZETA ---------------------
    temp_zeta = log(pow(param(j,1), 2));
    new_zeta  = arma::randn() * sqrt(accelerate_var(1)) + temp_zeta;
    tprob += - (q0_zeta) * new_zeta - q1_zeta / exp(new_zeta) + (q0_zeta) * temp_zeta + q1_zeta / exp(temp_zeta);
    new_zeta = sqrt(exp(new_zeta));
    temp_param(1) = new_zeta;
    
    // ----------- BETA ---------------------
    for(arma::uword l = 0; l < param.n_cols - 2; l++){
      temp_coef = param(j, l + 2);
      new_coef = arma::randn() * sqrt(accelerate_var(l + 2)) + temp_coef;
      temp_param(l + 2) = new_coef;
      
      tprob += log(arma::normpdf(new_coef, m_theta(l), sqrt(s2_theta(l)))) -
        log(arma::normpdf(temp_coef, m_theta(l), sqrt(s2_theta(l))));
      
    }
    
    // ----------- LIKELIHOOD ---------------
    if(model == 1){
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_typeI(y(i), temp_param(0), temp_param(1), temp_param(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= density_reg_typeI(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          } else {
            tprob += surv_reg_typeI(y(i), temp_param(0), temp_param(1), temp_param(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= surv_reg_typeI(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          }
        }
      }
    } else if(model == 2){
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_loglog(y(i), temp_param(0), temp_param(1), temp_param(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= density_reg_loglog(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          } else {
            tprob += surv_reg_loglog(y(i), temp_param(0), temp_param(1), temp_param(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= surv_reg_loglog(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          }
        }
      }
    } else if(model == 3){
      for(arma::uword i = 0; i < clust.n_elem; i++){
        if(clust(i) == j){
          if(cens(i) == 1){
            tprob += density_reg_norm(y(i), temp_param(0), temp_param(1), temp_param(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= density_reg_norm(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          } else {
            tprob += surv_reg_norm(y(i), temp_param(0), temp_param(1), temp_param(arma::span(2, cov_index)), covs.row(i).t());
            tprob -= surv_reg_norm(y(i), param(j,0), param(j,1), param(j,arma::span(2, cov_index)).t(), covs.row(i).t());
          }
        }
      }
    }
    
    if(log(arma::randu()) < std::min(0.0, tprob)){
      param.row(j) = temp_param.t();
    }
  }
}

//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
//------------------------------------ MAINS ---------------------------------------------------
//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List main_no_reg(arma::vec y,
                       arma::vec cens,
                       int niter, 
                       int nburn,
                       int napprox,
                       double m0, 
                       double s20, 
                       double q0_alpha, 
                       double q1_alpha, 
                       double s2v, 
                       double s2t,
                       double q0_tau, 
                       double q1_tau, 
                       double q0_zeta, 
                       double q1_zeta, 
                       arma::vec accelerate_var,
                       int model,
                       bool adapt = 1,
                       double alpha_u_star = 0.25,
                       bool output = 0){
  // initialize the results
  arma::mat clust_result(niter - nburn, y.n_elem);
  arma::vec alpha_result(niter - nburn);
  arma::vec u_result(niter - nburn);
  arma::vec tau_result(niter - nburn);
  arma::vec s2v_result(niter - nburn);
  arma::vec s2t_result(niter - nburn);
  arma::vec accelerate_count(niter, arma::fill::zeros);
  
  arma::mat mu_result(0,0);
  arma::mat zeta_result(0,0);
  if(output == 1){
    mu_result.resize(niter - nburn, y.n_elem);
    zeta_result.resize(niter - nburn, y.n_elem);
  }
  
  int acc_rate_u = 0;
  int acc_rate_tau = 0;
  
  // initialize the quantities
  arma::vec cl_unique;
  arma::mat param;
  arma::vec clust(y.n_elem);
  arma::vec temp_param_vec;
  
  double tau = 1;
  double alpha = 1;
  double u = 1;
  
  clust = arma::regspace(0, y.n_elem - 1);
  param.resize(y.n_elem, 2);
  param.col(0) = arma::randn(y.n_elem) * s20 + m0;
  param.col(1) = 1.0 / arma::randg(y.n_elem, arma::distr_param(q0_zeta, 1 / q1_zeta));
  
  // clust.fill(0);
  // param.resize(1,2);
  // param(0,0) = arma::randn() * s20 + m0;
  // param(0,1) = 1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0);
  
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  // main loop
  for(arma::uword iter = 0; iter < niter; iter++){
    
    // accelerate
    accelerate_no_reg(y, param, clust, cens, m0, s20, q0_zeta, q1_zeta, accelerate_var, accelerate_count, iter, model);
    cl_unique = unique(clust);
    
    // update alpha
    alpha = arma::randg(1, arma::distr_param(q0_alpha + cl_unique.n_elem, 1 / (q1_alpha + sqrt(u + tau) - sqrt(tau))))(0);
    
    // update u & tau
    u_update(u, s2v, alpha, tau, clust, acc_rate_u);  
    tau_update(tau, q0_tau, q1_tau, alpha, u, s2t, clust, acc_rate_tau);
    
    // update clusters
    clust_update_no_reg(napprox, m0, s20, q0_zeta, q1_zeta, u, tau, alpha, y, clust, cens, param, model);
    
    if(iter >= nburn){
      clust_result.row(iter - nburn) = clust.t();
      alpha_result(iter - nburn) = alpha;
      u_result(iter - nburn) = u;
      tau_result(iter - nburn) = tau;
      s2v_result(iter - nburn) = s2v;
      s2t_result(iter - nburn) = s2t;
      
      if(output == 1){
        temp_param_vec = param.col(0);
        mu_result.row(iter - nburn) = temp_param_vec.elem(arma::conv_to<arma::uvec>::from(clust)).t();
        temp_param_vec = param.col(1);
        zeta_result.row(iter - nburn) = temp_param_vec.elem(arma::conv_to<arma::uvec>::from(clust)).t();
      }
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
  results["alpha"] = alpha_result;
  results["u"] = u_result;
  results["tau"] = tau_result;
  results["s2v"] = s2v_result;
  results["s2t"] = s2t_result;
  results["acc_rate_u"] = acc_rate_u;
  results["acc_rate_tau"] = acc_rate_tau;
  results["accelerate_count"] = accelerate_count;
  results["time"] = double(current_s-start_s)/CLOCKS_PER_SEC;
  
  if(output == 1){
    results["mu"] = mu_result;
    results["zeta"] = zeta_result;
  }
  
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
                           double q0_alpha, 
                           double q1_alpha, 
                           double s2v, 
                           double s2t,
                           double q0_tau, 
                           double q1_tau, 
                           double q0_zeta, 
                           double q1_zeta, 
                           arma::vec accelerate_var,
                           int model,
                           bool adapt = 1,
                           double alpha_u_star = 0.25,
                           bool output = 0){
  
  // initialize the results
  arma::mat clust_result(niter - nburn, y.n_elem);
  arma::vec alpha_result(niter - nburn);
  arma::vec u_result(niter - nburn);
  arma::vec tau_result(niter - nburn);
  arma::vec s2v_result(niter - nburn);
  arma::vec s2t_result(niter - nburn);
  arma::vec accelerate_count(niter, arma::fill::zeros);
  
  arma::mat mu_result(0,0);
  arma::mat zeta_result(0,0);
  arma::mat reg_result(0,0);
  if(output == 1){
    mu_result.resize(niter - nburn, y.n_elem);
    zeta_result.resize(niter - nburn, y.n_elem);
    reg_result.resize(niter - nburn, covs.n_cols);
  }
  
  int acc_rate_u = 0;
  int acc_rate_tau = 0;
  
  // initialize the quantities
  arma::vec cl_unique;
  arma::mat param;
  arma::vec clust(y.n_elem);
  arma::vec common_reg(covs.n_cols);
  arma::vec temp_param_vec;
  
  double tau = 1;
  double alpha = 1;
  double u = 1;
  
  clust = arma::regspace(0, y.n_elem - 1);
  param.resize(y.n_elem, 2);
  param.col(0) = arma::randn(y.n_elem) * s20 + m0;
  param.col(1) = 1.0 / arma::randg(y.n_elem, arma::distr_param(q0_zeta, 1 / q1_zeta));
  common_reg.fill(0.0);
  
  // clust.fill(0);
  // param.resize(1,2);
  // param(0,0) = arma::randn() * s20 + m0;
  // param(0,1) = 1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0);
  // common_reg.fill(1.0);
  
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  // main loop
  for(arma::uword iter = 0; iter < niter; iter++){
    
    // accelerate
    accelerate_common_reg(y, covs, param, common_reg, clust, cens,  m0, s20, m_theta, s2_theta, 
                          q0_zeta, q1_zeta, accelerate_var, accelerate_count, iter, model);
    cl_unique = unique(clust);
    
    // update alpha
    alpha = arma::randg(1, arma::distr_param(q0_alpha + cl_unique.n_elem, 1 / (q1_alpha + sqrt(u + tau) - sqrt(tau))))(0);
    
    // update u & tau
    u_update(u, s2v, alpha, tau, clust, acc_rate_u);  
    tau_update(tau, q0_tau, q1_tau, alpha, u, s2t, clust, acc_rate_tau);
    
    // update clusters
    clust_update_common_reg(napprox, m0, s20, m_theta, s2_theta, q0_zeta, q1_zeta,
                            u, tau, alpha, y, clust, covs, cens, param, common_reg, model);
    
    if(iter >= nburn){
      clust_result.row(iter - nburn) = clust.t();
      alpha_result(iter - nburn) = alpha;
      u_result(iter - nburn) = u;
      tau_result(iter - nburn) = tau;
      s2v_result(iter - nburn) = tau;
      s2t_result(iter - nburn) = s2t;
      
      if(output == 1){
        temp_param_vec = param.col(0);
        mu_result.row(iter - nburn) = temp_param_vec.elem(arma::conv_to<arma::uvec>::from(clust)).t();
        temp_param_vec = param.col(1);
        zeta_result.row(iter - nburn) = temp_param_vec.elem(arma::conv_to<arma::uvec>::from(clust)).t();
        reg_result.row(iter - nburn) = common_reg.t();
      }
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
  results["alpha"] = alpha_result;
  results["u"] = u_result;
  results["tau"] = tau_result;
  results["s2v"] = s2v_result;
  results["s2t"] = s2t_result;
  results["acc_rate_u"] = acc_rate_u;
  results["acc_rate_tau"] = acc_rate_tau;
  results["accelerate_count"] = accelerate_count;
  results["time"] = double(current_s-start_s)/CLOCKS_PER_SEC;
  
  if(output == 1){
    results["mu"] = mu_result;
    results["zeta"] = zeta_result;
    results["reg"] = reg_result;
  }
  
  return results;
}

// [[Rcpp::export]]
Rcpp::List main_reg(arma::vec y, 
                    arma::mat covs,
                    arma::vec cens,
                    int niter, 
                    int nburn,
                    int napprox,
                    double m0, 
                    double s20, 
                    arma::vec m_theta,
                    arma::vec s2_theta,
                    double q0_alpha, 
                    double q1_alpha, 
                    double s2v, 
                    double s2t,
                    double q0_tau, 
                    double q1_tau, 
                    double q0_zeta, 
                    double q1_zeta, 
                    arma::vec accelerate_var,
                    int model,
                    bool adapt = 1,
                    double alpha_u_star = 0.25,
                    bool output = 0){
  
  // initialize the results
  arma::mat clust_result(niter - nburn, y.n_elem);
  arma::vec alpha_result(niter - nburn);
  arma::vec u_result(niter - nburn);
  arma::vec tau_result(niter - nburn);
  arma::vec s2v_result(niter - nburn);
  arma::vec s2t_result(niter - nburn);
  arma::vec accelerate_count(niter, arma::fill::zeros);
  
  arma::mat mu_result(0,0);
  arma::mat zeta_result(0,0);
  arma::cube reg_result(0,0,0);
  if(output == 1){
    mu_result.resize(niter - nburn, y.n_elem);
    zeta_result.resize(niter - nburn, y.n_elem);
    reg_result.resize(y.n_elem, covs.n_cols, niter - nburn);
  }
  
  int acc_rate_u = 0;
  int acc_rate_tau = 0;
  
  // initialize the quantities
  arma::vec cl_unique;
  arma::mat param;
  arma::vec clust(y.n_elem);
  arma::vec temp_param_vec;
  arma::mat temp_param_mat;
  
  double tau = 1;
  double alpha = 1;
  double u = 1;
  
  clust = arma::regspace(0, y.n_elem - 1);
  param.resize(y.n_elem, 2 + covs.n_cols);
  param.col(0) = arma::randn(y.n_elem) * s20 + m0;
  param.col(1) = 1.0 / arma::randg(y.n_elem, arma::distr_param(q0_zeta, 1 / q1_zeta));
  param.tail_cols(covs.n_cols).fill(0.0);
  
  // clust.fill(0);
  // param.resize(1,2 + covs.n_cols);
  // param(0,0) = arma::randn() * s20 + m0;
  // param(0,1) = 1.0 / arma::randg(1, arma::distr_param(q0_zeta, 1 / q1_zeta))(0);
  // param(0, arma::span(2, param.n_cols - 1)) = arma::randn(param.n_cols - 2).t() ;
  
  int start_s = clock();
  int current_s;
  int nupd = round(niter / 10);
  
  // main loop
  for(arma::uword iter = 0; iter < niter; iter++){
    
    // accelerate
    accelerate_reg(y, covs, param, clust, cens, m0, s20, m_theta, s2_theta, q0_zeta, q1_zeta, accelerate_var, accelerate_count, iter, model);
    cl_unique = unique(clust);
    
    // update alpha
    alpha = arma::randg(1, arma::distr_param(q0_alpha + cl_unique.n_elem, 1 / (q1_alpha + sqrt(u + tau) - sqrt(tau))))(0);
    
    // update u & tau
    u_update(u, s2v, alpha, tau, clust, acc_rate_u);  
    tau_update(tau, q0_tau, q1_tau, alpha, u, s2t, clust, acc_rate_tau);
    
    // update clusters
    clust_update_reg(napprox, m0, s20, m_theta, s2_theta, q0_zeta, q1_zeta, 
                     u, tau, alpha, y, clust, covs, cens, param, model);
    
    if(iter >= nburn){
      clust_result.row(iter - nburn) = clust.t();
      alpha_result(iter - nburn) = alpha;
      u_result(iter - nburn) = u;
      tau_result(iter - nburn) = tau;
      s2v_result(iter - nburn) = s2v;
      s2t_result(iter - nburn) = s2t;
      
      if(output == 1){
        temp_param_vec = param.col(0);
        mu_result.row(iter - nburn) = temp_param_vec.elem(arma::conv_to<arma::uvec>::from(clust)).t();
        temp_param_vec = param.col(1);
        zeta_result.row(iter - nburn) = temp_param_vec.elem(arma::conv_to<arma::uvec>::from(clust)).t();
        temp_param_mat = param.tail_cols(covs.n_cols);
        reg_result.slice(iter - nburn) = temp_param_mat.rows(arma::conv_to<arma::uvec>::from(clust));
      }
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
  results["alpha"] = alpha_result;
  results["u"] = u_result;
  results["tau"] = tau_result;
  results["s2v"] = s2v_result;
  results["s2t"] = s2t_result;
  results["acc_rate_u"] = acc_rate_u;
  results["acc_rate_tau"] = acc_rate_tau;
  results["accelerate_count"] = accelerate_count;
  results["time"] = double(current_s-start_s)/CLOCKS_PER_SEC;
  
  if(output == 1){
    results["mu"] = mu_result;
    results["zeta"] = zeta_result;
    results["reg"] = reg_result;
  }
  
  return results;
}
