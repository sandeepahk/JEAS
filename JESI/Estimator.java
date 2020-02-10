package jeas;

import java.io.File;

import org.apache.commons.math3.special.Gamma;


public class Estimator {
	// output model
		protected Model trnModel;
		LDACmdOption option;
		
		public boolean init(LDACmdOption option){
			this.option = option;
			trnModel = new Model();
			
			if (option.est){
				if (!trnModel.initNewModel(option))
					return false;
				trnModel.data.localDict.writeWordMap(option.dir + File.separator + option.wordMapFileName);
			}
			else if (option.estc){
				if (!trnModel.initEstimatedModel(option))
					return false;
			}
			
			return true;
		}
		
		public void estimate(){
			System.out.println("Sampling " + trnModel.niters + " iteration!");
			
			int lastIter = trnModel.liter;
			for (trnModel.liter = lastIter + 1; trnModel.liter < trnModel.niters + lastIter; trnModel.liter++){
				System.out.println("Iteration " + trnModel.liter + " ...");
				
				// for all z_i
				for (int m = 0; m < trnModel.M; m++){				
					for (int n = 0; n < trnModel.data.docs[m].length; n++){
						int [] results = new int [3];
						results = sampling(m, n);
						
						trnModel.e[m].set(n, results[0]);
						trnModel.t[m].set(n, results[1]);
						trnModel.s[m].set(n, results[2]);
					}// end for each word
					
				}// end for each document
				
				if (trnModel.updateParaSteps > 0 && trnModel.liter % trnModel.updateParaSteps == 0){
					//update_Prameters_Sentiment();
					
				}									
				
				if (option.savestep > 0){
					if (trnModel.liter % option.savestep == 0){
						System.out.println("Saving the model at iteration " + trnModel.liter + " ...");
						computeTheta();
						computePhi();
						trnModel.saveModel("model-" + Conversion.ZeroPad(trnModel.liter, 5));
					}
				}
			}// end iterations		
			
			System.out.println("Gibbs sampling completed!\n");
			System.out.println("Saving the final model!\n");
			computeTheta();
			computePhi();
			trnModel.liter--;
			trnModel.saveModel("model-final");
		}
		

		/**
		 * Do sampling
		 * @param m document number
		 * @param n word number
		 * @return topic id
		 */
		public int[] sampling(int m, int n){
			// remove z_i from the count variable
			int entity = trnModel.e[m].get(n);
			int topic = trnModel.t[m].get(n);
			int sentiment = trnModel.s[m].get(n);
			int w = trnModel.data.docs[m].words[n];
			
			trnModel.nwe[w][entity] -= 1;
			trnModel.nwt[w][topic] -= 1;
			trnModel.nws[w][sentiment] -= 1;
			trnModel.ne[entity] -= 1;
			trnModel.nt[topic] -= 1;
			trnModel.ns[sentiment] -= 1;
			trnModel.nde[m][entity] -= 1;
			trnModel.ndt[m][topic] -= 1;
			trnModel.ndets[m][entity][topic][sentiment] -= 1;
			trnModel.ndet[m][entity][topic] -= 1;
			trnModel.nd[m] -= 1;
		
			
			
			double Ealpha = trnModel.alpha_e * trnModel.E;
			double Talpha = trnModel.alpha_t * trnModel.T;
		
			
			
			//do multinominal sampling via cumulative method
			for (int e = 0; e < trnModel.E; e++){
				for (int t = 0; t < trnModel.T; t++) {
					for (int s = 0; s < trnModel.S; s++) {
						trnModel.p[e][t][s] = (trnModel.nwe[w][e] + trnModel.beta_ew[e][w])/(trnModel.ne[e] + trnModel.beta_esum[e]) *
								(trnModel.nwt[w][t] + trnModel.beta_tw[t][w])/(trnModel.nt[t] + trnModel.beta_tsum[t]) *
								(trnModel.nws[w][s] + trnModel.beta_sw[s][w])/(trnModel.ns[s] + trnModel.beta_ssum[s]) *
								(trnModel.nde[m][e] + trnModel.alpha_e)/(trnModel.nd[m] + Ealpha) *
								(trnModel.ndt[m][t] + trnModel.alpha_t)/(trnModel.nd[m] + Talpha) *
								(trnModel.ndets[m][e][t][s] + trnModel.alpha_ets[e][t][s])/(trnModel.ndet[m][e][t] + trnModel.alphaSum_et[e][t]);
						
					}
				}
				
			}
			
		
			// cumulate multinomial parameters
			for (int e = 0; e < trnModel.E; e++){
				for (int t = 0; t < trnModel.T; t++) {
					for (int s = 0; s < trnModel.S; s++) {
						if( s == 0) {
							if(t == 0) {
								if(e == 0) {
									continue;
								}
								else {
									trnModel.p[e][t][s] += trnModel.p[e-1][trnModel.T - 1][trnModel.S -1];
								}
							}
							else {
								trnModel.p[e][t][s] += trnModel.p[e][t-1][trnModel.S-1];
							}
							
							
						}
						else {
							
							trnModel.p[e][t][s] += trnModel.p[e][t][s-1];
						}
						
					}
				}
			}
			
			// scaled sample because of unnormalized p[]
			double u = Math.random() * trnModel.p[trnModel.E - 1][trnModel.T - 1][trnModel.S - 1];
			
			boolean loopbreak = false;
			for (entity = 0; entity < trnModel.E ; entity++){
				for (topic = 0; topic < trnModel.T ; topic++) {
					for (sentiment = 0; sentiment < trnModel.S; sentiment++) {
						if (trnModel.p[entity][topic][sentiment] > u) {//sample topic w.r.t distribution p
							loopbreak = true;
							break;
						}
							
					}
					
					if(loopbreak) {
						break;
					}
				}
				
				if(loopbreak) {
					break;
				}
			}
			
			/*
			if( entity == trnModel.E) entity--;
			if ( topic == trnModel.T) topic--;
			if ( sentiment == trnModel.S) sentiment--;*/
			// add newly estimated z_i to count variables
			
			trnModel.nwe[w][entity] += 1;
			trnModel.nwt[w][topic] += 1;
			trnModel.nws[w][sentiment] += 1;
			trnModel.ne[entity] += 1;
			trnModel.nt[topic] += 1;
			trnModel.ns[sentiment] += 1;
			trnModel.nde[m][entity] += 1;
			trnModel.ndt[m][topic] += 1;
			trnModel.ndets[m][entity][topic][sentiment] += 1;
			trnModel.ndet[m][entity][topic] += 1;
			trnModel.nd[m] += 1;
			
			int [] results = new int [3]; 
			
			results[0] = entity;
			results[1] = topic;
			results[2] = sentiment;
			
	 		return results;
		}
		
		public void computeTheta(){
			for (int m = 0; m < trnModel.M; m++){
				for (int e = 0; e < trnModel.E; e++){
					trnModel.theta_e[m][e] = (trnModel.nde[m][e] + trnModel.alpha_e) / (trnModel.nd[m] + trnModel.E * trnModel.alpha_e);
				}
			}
			for (int m = 0; m < trnModel.M; m++){
				for (int t = 0; t < trnModel.T; t++){
					trnModel.theta_t[m][t] = (trnModel.ndt[m][t] + trnModel.alpha_t) / (trnModel.nd[m] + trnModel.T * trnModel.alpha_t);
				}
			}
			
			for (int m = 0; m < trnModel.M; m++){
				for (int e = 0; e < trnModel.E; e++){
					for (int t = 0; t < trnModel.T; t++){
						for (int s = 0; s < trnModel.S; s++) {
							trnModel.theta_s[m][e][t][s] = (trnModel.ndets[m][e][t][s] + trnModel.alpha_ets[e][t][s]) / (trnModel.ndet[m][e][t] + trnModel.alphaSum_et[e][t]);
						}
					}
				}
			}
			
			
		}
		
		public void computePhi(){
			for (int e = 0; e < trnModel.E; e++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_e[e][w] = (trnModel.nwe[w][e] + trnModel.beta_ew[e][w]) / (trnModel.ne[e] + trnModel.beta_esum[e]);
				}
			}
			
			for (int t = 0; t < trnModel.T; t++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_t[t][w] = (trnModel.nwt[w][t] + trnModel.beta_tw[t][w]) / (trnModel.nt[t] + trnModel.beta_tsum[t]);
				}
			}
			
			for (int s = 0; s < trnModel.S; s++){
				for (int w = 0; w < trnModel.V; w++){
					trnModel.phi_s[s][w] = (trnModel.nws[w][s] + trnModel.beta_sw[s][w]) / (trnModel.ns[s] + trnModel.beta_ssum[s]);
				}
			}
		}
		
		public int update_Prameters_Sentiment(){
			int[][] data = new int[trnModel.S][trnModel.M];
			double[] alpha_temp = new double[trnModel.S];
			
			for (int s = 0; s < trnModel.S; s++){
				for (int m = 0; m < trnModel.M; m++){
					data[s][m] = 0;
				}
				
			}
			
			for (int s = 0; s < trnModel.S; s++){
				alpha_temp[s] = 0;
			}
			
			//update alpha
			for(int e = 0; e < trnModel.E; e++){
				for(int t = 0; t < trnModel.T; t++) {
					
					for(int s = 0; s < trnModel.S; s++){
						for(int m = 0; m <trnModel.M; m++){
							
							data[s][m] += trnModel.ndets[m][e][t][s];
						}
					}
				
					for(int s = 0; s < trnModel.S; s++){
						alpha_temp[s] = trnModel.alpha_ets[e][t][s];
					}
				}
			}
			
			polya_fit_simple(data, alpha_temp, trnModel.S, trnModel.M);
			
			for(int e = 0; e < trnModel.E; e++){
				for(int t = 0; t < trnModel.T; t++) {
					// update alpha
					trnModel.alphaSum_et[e][t] = 0.0;
					for (int s = 0; s < trnModel.S; s++) {
						trnModel.alpha_ets[e][t][s] = alpha_temp[s];
						trnModel.alphaSum_et[e][t] += trnModel.alpha_ets[e][t][s];
					}
				}
			}
					
					
					
			
			return 0;
			
		}
		
		private int polya_fit_simple(int[][] data, double[] alpha, int _K, int _nSample ){
			int K = _K;
			int nSample = _nSample;
			int polya_iter = Integer.MAX_VALUE;
			
			
			double sum_alpha_old;
			double[] old_alpha = new double[K];
			double sum_g = 0; //sum_g = sum_digama(data[i][k] + old_alpha[k]),
			double sum_h = 0; // sum_h + sum_digama(data[i] + sum_alpha_old) , where data[i] = sum_data[i][k] for all k,
			double[] data_row_sum = new double[nSample]; // the sum of the counts of each data sample P = {P_1, P_2,...,P_k}
			boolean sat_state = false;
			int i, k, j;
			
			for (k = 0; k < K; k++){
				old_alpha[k] = 0;
			}
			
			for (i = 0; i < nSample; i++) {
				data_row_sum[i] = 0;
			}
			
			for (i = 0; i < nSample; i++) {
				for (k = 0; k < K; k++){
					
					data_row_sum[i] += data[k][i];
					
				}
				
				
			}
			
			//simplw fix point interation
			System.out.println("Optimising parameters....");
			for(i = 0; i < polya_iter; i++){
				sum_alpha_old = 0;
				// update old_alpha after each iteration
				for(j = 0; j < K; j++){
					old_alpha[j] = alpha[j];
				}
				
				//calculate sum_alpha_old
				for (j = 0; j < K; j++) {
					 sum_alpha_old += old_alpha[j];
				 }
				
				for (k = 0; k < K; k++) {
					 sum_g = 0;
					 sum_h = 0;
					 
					 // calculate sum_g[k]
					 for (j = 0; j < nSample; j++) {
						 sum_g += Gamma.digamma( data[k][j] + old_alpha[k]);
					 }
					 
					// calculate sum_h
					 for (j = 0; j < nSample; j++) {
						 sum_h += Gamma.digamma(data_row_sum[j] + sum_alpha_old);
					 }
					 
							
					// update alpha (new)
					double x = sum_g - nSample * Gamma.digamma(old_alpha[k]);
					double y = sum_h - nSample * Gamma.digamma(sum_alpha_old);
					
					alpha[k] = old_alpha[k] * (x / y);
					
					
					
					
					 
						
					 
				}
				
				 // terminate iteration ONLY if each dimension of {alpha_1, alpha_2, ... alpha_k} satisfy the termination criteria,
				 for (j = 0; j < K; j++) {
					 if (Math.abs( alpha[j] - old_alpha[j]) > 0.00000001 ) {
						 break;
					 }
						 
					 if ( j == K - 1) {
						 sat_state = true;
					 }
				 }
				 
				// check whether to terminate the whole iteration
					if(sat_state) {
						System.out.println("Terminated at iteration: "+i);
						break;
					}
					else if(i == polya_iter-1) {
						
						System.out.println("Haven't converged! Terminated at iteration: "+(i+1));
					}
						
	 

				
			}
			 for (k = 0; k < K; k++) {
				 System.out.print(alpha[k] + "\t");
				 
			 }
			
			
			System.out.println("Optimisation done!");

		    return 0;
		}
		
		

}
