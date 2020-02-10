package jeas_reverse;

import java.io.File;

public class Inferencer {
	// Train model
		public Model trnModel;
		public Dictionary globalDict;
		private LDACmdOption option;
		
		private Model newModel;
		//public int niters = 100;

	//-----------------------------------------------------
	// Init method
	//-----------------------------------------------------
	public boolean init(LDACmdOption option){
		this.option = option;
		trnModel = new Model();
			
		if (!trnModel.initEstimatedModel(option))
			return false;		
			
		globalDict = trnModel.data.localDict;
		computeTrnTheta();
		computeTrnPhi();
		
			
		return true;
	}
	
	protected void computeTrnTheta(){
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
					trnModel.theta_t[m][s][t] = (trnModel.ndst[m][s][t] + trnModel.alpha_st[s][t]) / (trnModel.nds[m][s] + trnModel.T * trnModel.alphaSum_st[s]);
				}
			}
			
		}
		
		for (int m = 0; m < trnModel.M; m++){
				for (int s = 0; s < trnModel.S; s++) {
						trnModel.theta_s[m][s] = (trnModel.nds[m][s] + trnModel.alpha_s) / (trnModel.nd[m] + trnModel.alpha_s * trnModel.S);
					}
		}
	}
	
	protected void computeTrnPhi(){
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
	
	//inference new model ~ getting dataset from file specified in option
		public Model inference(){	
			//System.out.println("inference");
			
			newModel = new Model();
			if (!newModel.initNewModel(option, trnModel)) return null;
			
			trnModel.data.localDict.writeWordMap(option.dir + File.separator + "new"+option.wordMapFileName);
			
			System.out.println("Sampling " + newModel.niters + " iteration for inference!");
			
			for (newModel.liter = 1; newModel.liter <= newModel.niters; newModel.liter++){
				System.out.println("Iteration " + newModel.liter + " ...");
				
				// for all newz_i
				for (int m = 0; m < newModel.M; ++m){
					for (int n = 0; n < newModel.data.docs[m].length; n++){
						
						
						int [] results = new int [3];
						results = infSampling(m, n);
						
						newModel.e[m].set(n, results[0]);
						newModel.t[m].set(n, results[1]);
						newModel.s[m].set(n, results[2]);
					}
				}//end foreach new doc
				
			}// end iterations
			
			System.out.println("Gibbs sampling for inference completed!");		
			System.out.println("Saving the inference outputs!");
			
			computeNewTheta();
			computeNewPhi();
			newModel.liter--;
			newModel.saveModel(newModel.dfile + "." + newModel.modelName);		
			
			return newModel;
		}
		
		/**
		 * do sampling for inference
		 * m: document number
		 * n: word number?
		 */
		protected int[] infSampling(int m, int n){
			// remove z_i from the count variables
			int entity = newModel.e[m].get(n);
			int topic = newModel.t[m].get(n);
			int sentiment = newModel.s[m].get(n);
			int _w = newModel.data.docs[m].words[n];
			int w = newModel.data.lid2gid.get(_w);
			


			newModel.nwe[_w][entity] -= 1;
			newModel.nwt[_w][topic] -= 1;
			newModel.nws[_w][sentiment] -= 1;
			newModel.ne[entity] -= 1;
			newModel.nt[topic] -= 1;
			newModel.ns[sentiment] -= 1;
			newModel.ndse[m][sentiment][entity] -= 1;
			newModel.ndst[m][sentiment][topic] -= 1;
			newModel.nds[m][sentiment] -= 1;
			newModel.nd[m] -= 1;
			
			
			
			double Salpha = newModel.alpha_s * newModel.S;
			
		
			
			
				//do multinominal sampling via cumulative method
				for (int e = 0; e < newModel.E; e++){
					for (int t = 0; t < newModel.T; t++) {
						for (int s = 0; s < newModel.S; s++) {
							newModel.p[e][t][s] = (trnModel.nwe[w][e] + newModel.nwe[_w][e] + newModel.beta_ew[e][_w])/(trnModel.ne[e] + newModel.ne[e] + newModel.beta_esum[e]) *
									(trnModel.nwt[w][t] + newModel.nwt[_w][t] + newModel.beta_tw[t][_w])/(trnModel.nt[t]  + newModel.nt[t] + newModel.beta_tsum[t]) *
									(trnModel.nws[w][s] + newModel.nws[_w][s] + newModel.beta_sw[s][_w])/(trnModel.ns[s] + newModel.ns[s] + newModel.beta_ssum[s]) *
									(newModel.ndse[m][s][e] + newModel.alpha_se[s][e])/(newModel.nds[m][s] + newModel.alphaSum_se[s]) *
									(newModel.ndst[m][s][t] + newModel.alpha_st[s][t])/(newModel.nds[m][s] + newModel.alphaSum_st[s]) *
									(newModel.nds[m][s] + newModel.alpha_s)/(newModel.nd[m] + Salpha);
						}
					}
					
				}
			
				
			
			
			
			
			
			// cumulate multinomial parameters
			for (int e = 0; e < newModel.E; e++){
				for (int t = 0; t < newModel.T; t++) {
					for (int s = 0; s < newModel.S; s++) {
						if( s == 0) {
							if(t == 0) {
								if(e == 0) {
									continue;
								}
								else {
									newModel.p[e][t][s] += newModel.p[e - 1][newModel.T - 1][newModel.S - 1];
								}
							}
							else {
								newModel.p[e][t][s] += newModel.p[e][t - 1][newModel.S - 1];
							}
							
							
						}
						else {
							
							newModel.p[e][t][s] += newModel.p[e][t][s - 1];
						}
						
					}
				}
			}
			
			// scaled sample because of unnormalized p[]
			double u = Math.random() * newModel.p[newModel.E - 1][newModel.T - 1][newModel.S - 1];
			
			boolean loopbreak = false;
			for (entity = 0; entity < newModel.E ; entity++){
				for (topic = 0; topic < newModel.T ; topic++) {
					for (sentiment = 0; sentiment < newModel.S; sentiment++) {
						if (newModel.p[entity][topic][sentiment] > u) {//sample topic w.r.t distribution p
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
				
			if( entity == newModel.E) entity--;
			if ( topic == newModel.T) topic--;
			if ( sentiment == newModel.S) sentiment--;
			
			// add newly estimated z_i to count variables
			
			newModel.nwe[_w][entity] += 1;
			newModel.nwt[_w][topic] += 1;
			newModel.nws[_w][sentiment] += 1;
			newModel.ne[entity] += 1;
			newModel.nt[topic] += 1;
			newModel.ns[sentiment] += 1;
			newModel.ndse[m][sentiment][entity] += 1;
			newModel.ndst[m][sentiment][topic] += 1;
			newModel.nds[m][sentiment] += 1;
			newModel.nd[m] += 1;
			
			int [] results = new int [3]; 
			
			results[0] = entity;
			results[1] = topic;
			results[2] = sentiment;
			
	 		return results;
			

		}
		
		protected void computeNewTheta(){
			for (int e = 0; e < newModel.E; e++){
				for (int w = 0; w < newModel.V; w++){
					newModel.phi_e[e][w] = (newModel.nwe[w][e] + newModel.beta_ew[e][w]) / (newModel.ne[e] + newModel.beta_esum[e]);
				}
			}
			
			for (int t = 0; t < newModel.T; t++){
				for (int w = 0; w < newModel.V; w++){
					newModel.phi_t[t][w] = (newModel.nwt[w][t] + newModel.beta_tw[t][w]) / (newModel.nt[t] + newModel.beta_tsum[t]);
				}
			}
			
			for (int s = 0; s < newModel.S; s++){
				for (int w = 0; w < newModel.V; w++){
					newModel.phi_s[s][w] = (newModel.nws[w][s] + newModel.beta_sw[s][w]) / (newModel.ns[s] + newModel.beta_ssum[s]);
				}
			}
		}
		
		protected void computeNewPhi(){
		
			
			for (int e = 0; e < newModel.E; e++){
				for (int w = 0; w < newModel.V; w++){
					
						newModel.phi_e[e][w] = (trnModel.nwe[w][e] + newModel.nwe[w][e] + newModel.beta_ew[e][w]) / (trnModel.ne[e] + newModel.ne[e] + newModel.beta_esum[e]);
					
					
				}
			}
			
			for (int t = 0; t < newModel.T; t++){
				for (int w = 0; w < newModel.V; w++){
					
						newModel.phi_t[t][w] = (trnModel.nwt[w][t] +newModel.nwt[w][t] + newModel.beta_tw[t][w]) / (trnModel.nt[t] + newModel.nt[t] + newModel.beta_tsum[t]);
					
					
				}
			}
			
			for (int s = 0; s < newModel.S; s++){
				for (int w = 0; w < newModel.V; w++){
					
						newModel.phi_s[s][w] = (trnModel.nws[w][s] +newModel.nws[w][s] + newModel.beta_sw[s][w]) / (trnModel.ns[s] + newModel.ns[s] + newModel.beta_ssum[s]);
					
					
				}
			}
		}
		
}
