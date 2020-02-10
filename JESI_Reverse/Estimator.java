package jeas_reverse;

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
			trnModel.ndse[m][sentiment][entity] -= 1;
			trnModel.ndst[m][sentiment][topic] -= 1;
			trnModel.nds[m][sentiment] -= 1;
			trnModel.nd[m] -= 1;
		
			
			
			double Salpha = trnModel.alpha_s * trnModel.S;
			
		
			
			
			//do multinominal sampling via cumulative method
			for (int e = 0; e < trnModel.E; e++){
				for (int t = 0; t < trnModel.T; t++) {
					for (int s = 0; s < trnModel.S; s++) {
						trnModel.p[e][t][s] = (trnModel.nwe[w][e] + trnModel.beta_ew[e][w])/(trnModel.ne[e] + trnModel.beta_esum[e]) *
								(trnModel.nwt[w][t] + trnModel.beta_tw[t][w])/(trnModel.nt[t] + trnModel.beta_tsum[t]) *
								(trnModel.nws[w][s] + trnModel.beta_sw[s][w])/(trnModel.ns[s] + trnModel.beta_ssum[s]) *
								(trnModel.ndse[m][s][e] + trnModel.alpha_se[s][e])/(trnModel.nds[m][s] + trnModel.alphaSum_se[s]) *
								(trnModel.ndst[m][s][t] + trnModel.alpha_st[s][t])/(trnModel.nds[m][s] + trnModel.alphaSum_st[s]) *
								(trnModel.nds[m][s] + trnModel.alpha_s)/(trnModel.nd[m] + Salpha);
						
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
			trnModel.ndse[m][sentiment][entity] += 1;
			trnModel.ndst[m][sentiment][topic] += 1;
			trnModel.nds[m][sentiment] += 1;
			trnModel.nd[m] += 1;
			
			int [] results = new int [3]; 
			
			results[0] = entity;
			results[1] = topic;
			results[2] = sentiment;
			
	 		return results;
		}
		
		public void computeTheta(){
			for (int m = 0; m < trnModel.M; m++){
				for (int s = 0; s < trnModel.S; s++) {
					for (int e = 0; e < trnModel.E; e++){
						trnModel.theta_e[m][s][e] = (trnModel.ndse[m][s][e] + trnModel.alpha_se[s][e]) / (trnModel.nds[m][s] + trnModel.alphaSum_se[s]);
					}
				}
				
			}
			for (int m = 0; m < trnModel.M; m++){
				for (int s = 0; s < trnModel.S; s++) {
					for (int t = 0; t < trnModel.T; t++){
						trnModel.theta_t[m][s][t] = (trnModel.ndst[m][s][t] + trnModel.alpha_st[s][t]) / (trnModel.nds[m][s] + trnModel.alphaSum_st[s]);
					}
				}
				
			}
			
			for (int m = 0; m < trnModel.M; m++){
					for (int s = 0; s < trnModel.S; s++) {
							trnModel.theta_s[m][s] = (trnModel.nds[m][s] + trnModel.alpha_s) / (trnModel.nd[m] + trnModel.alpha_s * trnModel.S);
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
		
		

}
