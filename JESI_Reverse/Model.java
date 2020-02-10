/*
 * Written by: Sandeepa Kannangara, University of New South Wales, s.kannangara@unsw.edu.au
 * Part of code is from http://gibbslda.sourceforge.net/.
*/

package jeas_reverse;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Vector;

import uk.ac.wlv.sentistrength.SentiStrength;

public class Model {
	
	//---------------------------------------------------------------
	//	Class Variables
	//---------------------------------------------------------------
	
	public static String tassignSuffix;	//suffix for topic assignment file
	public static String thetaSuffix;		//suffix for theta (topic - document distribution) file
	public static String phiSuffix;		//suffix for phi file (topic - word distribution) file
	public static String othersSuffix; 	//suffix for containing other parameters
	public static String twordsSuffix;		//suffix for file containing words-per-topics
	
	//---------------------------------------------------------------
	//	Model Parameters and Variables
	//---------------------------------------------------------------
	
	public String wordMapFile; 		//file that contain word to id map
	public String trainlogFile; 	//training log file	
	
	public String dir;
	public String dfile;
	public String modelName;
	public int modelStatus; 		//see Constants class for status of model
	public LDADataset data;			// link to a dataset
	
	public int M; //dataset size (i.e., number of docs)
	public int V; //vocabulary size
	public int E; //number of entities
	public int S; //number of sentiment
	public int T; //number of topics
	
	// temp variables for sampling
	protected double [][][] p; //size E x T x S
	
	public double alpha_e, alpha_s, alpha_t, beta_e, beta_s, beta_t; //LDA  hyperparameters
	public double [][] alpha_se;
	public double [] alphaSum_se;
	public double [][] alpha_st;
	public double [] alphaSum_st;
	public double [][] beta_ew, beta_sw, beta_tw; //size E x V, S x v, T x V
	public double [] beta_esum, beta_ssum, beta_tsum; //sum of beta values
	public double [][] lambda_e, lambda_s, lambda_t; //size E x V, S x V, T x v---for encoding prior topic information 
	public int niters; //number of Gibbs sampling iteration
	public int liter; //the iteration at which the model was saved	
	public int savestep; //saving period
	public int twords; //print out top words per each topic
	public int updateParaSteps;// prameter update steps
	
	// Estimated/Inferenced parameters
	public double [][][] theta_e, theta_t; //theta: document - topic distributions, size M x E, M x T
	public double [][] theta_s; //size M x S x E x T
	public double [][] phi_e, phi_s, phi_t; // phi: topic-word distributions, size E x V, S x V, T x V
	
	// Temp variables while sampling
	public Vector<Integer> [] e; //entity assignments for words, size M x doc.size()
	public Vector<Integer> [] s; //sentiment assignments for words, size M x doc.size()
	public Vector<Integer> [] t; //topic assignments for words, size M x doc.size()
	protected int [][] nwe, nws, nwt; //nw[i][j]: number of instances of word/term i assigned to topic j, size V x E, V x S, V x T
	protected int [][][] ndse, ndst; //nd[i][j]: number of words in document i assigned to topic j, size M x E,  M x T
	protected int [] ne, ns, nt; //nwsum[j]: total number of words assigned to topic j, size E, S, T
	protected int [] nd; //ndsum[i]: total number of words in document i, size M
	protected int [][] nds; //ndet_s[i][j][k][l]: number of words in document i assigned to entity j of topic k of sentiment l,  size M x E x T x S
	
	
	private ArrayList<List<String>> initialEntityWord = new ArrayList<List<String>>();
	private ArrayList<List<String>> initialTopicWord = new ArrayList<List<String>>();
	private ArrayList<List<String>> initialSentimentWord = new ArrayList<List<String>>();
	
	public Model(){
		setDefaultValues();	
	}
	
	/**
	 * Set default values for variables
	 */
	public void setDefaultValues(){
		wordMapFile = "wordmap.txt";
		trainlogFile = "trainlog.txt";
		tassignSuffix = ".tassign";
		thetaSuffix = ".theta";
		phiSuffix = ".phi";
		othersSuffix = ".others";
		twordsSuffix = ".twords";
		
		dir = "./";
		dfile = "trndocs.dat";
		modelName = "model-final";
		modelStatus = Constants.MODEL_STATUS_UNKNOWN;		
		
		M = 0;
		V = 0;
		E = 100;
		S = 100;
		T = 100;
		alpha_e = 50.0 / E;
		alpha_s = 50.0 / S;
		alpha_t = 50.0 / T;
		beta_e = 0.1;
		beta_s = 0.1;
		beta_t = 0.1;
		niters = 2000;
		liter = 0;
		updateParaSteps = 40;
		
		e = null;
		s = null;
		t = null;
		nwe = null;
		nws = null;
		nwt = null;
		ndse = null;
		ndst = null;
		ne = null;
		ns = null;
		nt = null;
		nd = null;
		nds = null;
		
		
		theta_e = null;
		theta_s = null;
		theta_t = null;
		phi_e = null;
		phi_s = null;
		phi_t = null;
	}
	
	/**
	 * Init parameters for estimation
	 */
	public boolean initNewModel(LDACmdOption option){
		if (!init(option))
			return false;
		
		p = new double[E][T][S];		
		
		data = LDADataset.readDataSet(dir + File.separator + dfile);
		if (data == null){
			System.out.println("Fail to read training data!\n");
			return false;
		}
		
		//+ allocate memory and assign values for variables		
		M = data.M;
		V = data.V;
		//dir = option.dir;
		savestep = option.savestep;
		updateParaSteps = option.updateParaSteps;
		
		// K: from command line or default value
	    // alpha, beta: from command line or default values
	    // niters, savestep: from command line or default values

		nwe = new int[V][E];
		for (int w = 0; w < V; w++){
			for (int e = 0; e < E; e++){
				nwe[w][e] = 0;
			}
		}
		
		nwt = new int[V][T];
		for (int w = 0; w < V; w++){
			for (int t = 0; t < T; t++){
				nwt[w][t] = 0;
			}
		}
		
		nws = new int[V][S];
		for (int w = 0; w < V; w++){
			for (int s = 0; s < S; s++){
				nws[w][s] = 0;
			}
		}
		
		ne =  new int[E];
		for (int e = 0; e < E; e++) {
			ne[e] = 0;
		}
		
		nt =  new int[T];
		for (int t = 0; t < T; t++) {
			nt[t] = 0;
		}
		
		ns =  new int[S];
		for (int s = 0; s < S; s++) {
			ns[s] = 0;
		}
		
		ndse = new int[M][S][E];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				for (int e = 0; e < E; e++) {
					ndse[m][s][e] = 0;
				}
			}			
		}
		
		ndst = new int[M][S][T];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				for (int t = 0; t < T; t++) {
					ndst[m][s][t] = 0;
				}
			}			
		}
		
		nd = new int[M];
		for (int m = 0; m < M; m++){
			nd[m] = 0;
			
		}
		
		nds =  new int[M][S];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				nds[m][s] = 0;
			}
					
			
		}
		
			
		loadData();
		
		e = new Vector[M];
		t = new Vector[M];
		s = new Vector[M];
		
		SentiStrength sentiStrength = new SentiStrength(); 
		String ssthInitialisation[] = {"sentidata", "TopicLists/sentiment/SentiStrength_DataEnglishFeb2017/"};
		sentiStrength.initialiseAndRun(ssthInitialisation); 
		/*
		Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		
       */
        
		for (int m = 0; m < data.M; m++){
			int N = data.docs[m].length;
			e[m] = new Vector<Integer>();
			t[m] = new Vector<Integer>();
			s[m] = new Vector<Integer>();
			
			StringTokenizer tknr = new StringTokenizer(data.docs[m].rawStr, " \t\r\n");
			
			//initilize for e, t and s
			for (int n = 0; n < N; n++){
				String token = tknr.nextToken();
				Integer entity = null;
				Integer topic = null;
				Integer sentiment = null;
				int mainSentiment = 0;
				int longest = 0;
				
				int i = 0;
				for(List<String> entityList:initialEntityWord){
					
					for(String entityWord:entityList){
						if(token.equals(entityWord.toLowerCase())){
							entity = i;
							break;
						}
					}
					i++;
					if(entity != null){
						break;
					}
				}
				
				i = 0;
				for(List<String> topicList:initialTopicWord){
					
					for(String topicWord:topicList){
						if(token.equals(topicWord.toLowerCase())){
							topic = i;
							break;
						}
					}
					i++;
					if(topic != null){
						break;
					}
				}
				
				
				String[] words = sentiStrength.computeSentimentScores(token).split("\\s+");
				int pos = Integer.parseInt(words[0]);
				int neg = Integer.parseInt(words[1]);
				//int sen = Integer.parseInt(words[2]);
				
				if (Math.abs(pos) < Math.abs(neg)) {
					sentiment = 1;
				}else if(Math.abs(pos) > Math.abs(neg)) {
					sentiment = 0;
				} 
				/*
				Annotation annotation = pipeline.process(token);
				for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
		            Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
		            int sentiment_ = RNNCoreAnnotations.getPredictedClass(tree);
		            String partText = sentence.toString();
		            if (partText.length() > longest) {
		                mainSentiment = sentiment_;
		                longest = partText.length();
		            }

		        }
				
				if(mainSentiment == 0 || mainSentiment == 1) {
					sentiment = 1;
				} else if (mainSentiment == 3 || mainSentiment == 4) {
					sentiment  = 0;
				}
				*/
				if(entity == null){
					entity = (int)Math.floor(Math.random() * E);
				}
				if(topic == null){
					topic = (int)Math.floor(Math.random() * T);
				}
				if(sentiment == null){
					sentiment = (int)Math.floor(Math.random() * S);
				}
				
				e[m].add(entity);
				t[m].add(topic);
				s[m].add(sentiment);
				
				nwe[data.docs[m].words[n]][entity] += 1;
				nwt[data.docs[m].words[n]][topic] += 1;
				nws[data.docs[m].words[n]][sentiment] += 1;
				ne[entity] += 1;
				nt[topic] += 1;
				ns[sentiment] += 1;
				ndse[m][sentiment][entity] += 1;
				ndst[m][sentiment][topic] += 1;
				nds[m][sentiment] += 1;
				nd[m] += 1;
			}
			
			
			//System.out.println(e[m]);
			
		}
		
		
		
		theta_e = new double[M][S][E];
		theta_t = new double[M][S][T];
		theta_s = new double[M][S];
		
		phi_e = new double[E][V];
		phi_t = new double[T][V];
		phi_s = new double[S][V];
		
		//initialize beta_ew with value of beta
		beta_ew = new double[E][V];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				beta_ew[e][w] = beta_e;
			}
		}
		
		//initialize beta_ew with value of beta
		beta_tw = new double[T][V];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				beta_tw[t][w] = beta_t;
			}
		}
		
		//initialize beta_ew with value of beta
		beta_sw=new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w] = beta_s;
			}
		}
				
		//Initialize lambda_e 
		lambda_e = new double[E][V];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				lambda_e[e][w] = 1;
			}
		}
		
		//Initialize lambda_t 
		lambda_t = new double[T][V];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				lambda_t[t][w] = 1;
			}
		}
		
		//Initialize lambda_s 
		lambda_s = new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				lambda_s[s][w] = 1;
			}
		}
		
		alpha_se = new double[S][E];
		alphaSum_se = new double[S];
		
		for(int s = 0; s < S; s++){	
				alphaSum_se[s] = 0;
				for(int e = 0; e < E; e++){
					
					alpha_se[s][e] = alpha_e;
					alphaSum_se[s] += alpha_se[s][e];
				}
			
		}
		
		alpha_st = new double[S][T];
		alphaSum_st = new double[S];
		
		
		for(int s = 0; s < S; s++){	
				alphaSum_st[s] = 0;
				for(int t = 0; t < T; t++){
					
					alpha_st[s][t] = alpha_t;
					alphaSum_st[s] += alpha_st[s][t];
				}
			
		}
		
		prior2beta();
		
		

		beta_esum = new double[E];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				beta_esum[e] += beta_ew[e][w];
			}
		}
		
		beta_tsum = new double[T];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				beta_tsum[t] += beta_tw[t][w];
			}
		}
		
		beta_ssum = new double[S];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_ssum[s] += beta_sw[s][w];
			}
		}
		return true;
	}
	
	/**
	 * initialize the model
	 */
	protected boolean init(LDACmdOption option){		
		if (option == null)
			return false;
		
		modelName = option.modelName;
		E = option.E;
		S = option.S;
		T = option.T;
		
		alpha_e = option.alpha_e;
		alpha_s = option.alpha_s;
		alpha_t = option.alpha_t;
		
		if (alpha_e < 0.0)
			alpha_e = 50.0 / E;
		
		if (alpha_s < 0.0)
			alpha_s = 50.0 / S;
		
		if (alpha_t < 0.0)
			alpha_t = 50.0 / T;
				
		if (option.beta_e >= 0)
			beta_e = option.beta_e;
		
		if (option.beta_s >= 0)
			beta_s = option.beta_s;
		
		if (option.beta_t >= 0)
			beta_t = option.beta_t;
		
		niters = option.niters;
		
		dir = option.dir;
		if (dir.endsWith(File.separator))
			dir = dir.substring(0, dir.length() - 1);
		
		dfile = option.dfile;
		twords = option.twords;
		wordMapFile = option.wordMapFileName;
		
		return true;
	}
	
	public void loadData() {
		
		initialEntityWord.add(loadPriorData("TopicLists/entity/10001.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10002.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10003.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10004.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10005.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10006.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10007.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10008.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10009.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10010.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10011.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10013.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10014.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10015.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10016.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10017.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10018.txt"));
		initialEntityWord.add(loadPriorData("TopicLists/entity/10019.txt"));
		
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10000.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10001.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10003.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10005.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10006.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10007.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10008.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10009.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10010.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10012.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10013.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10014.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10015.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10016.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10017.txt"));
		initialTopicWord.add(loadPriorData("TopicLists/newTopic/10018.txt"));
		
		
		//initialSentimentWord.add(loadPriorData("TopicLists/sentiment/positiveNRC.txt"));
		//initialSentimentWord.add(loadPriorData("TopicLists/sentiment/negativeNRC.txt"));
		
		//loadPriorDataMPQA("TopicLists/sentiment/MPQA.txt");
		
	}

	public List<String> loadPriorData(String topiclist){
		List topics=new ArrayList();
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(topiclist), "UTF-8"));
			String line = null;
			while ((line = reader.readLine()) != null) {
				topics.add(line);
				
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return topics;
		
	}
	
	public void loadPriorDataMPQA(String topiclist){
		List<String> posTopics=new ArrayList<String>();
		List<String> negTopics=new ArrayList<String>();
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(topiclist), "UTF-8"));
			String line = null;
			while ((line = reader.readLine()) != null) {
				String[] values = line.split("\\s+");
				
				if(values[2].equals("0.9") && !negTopics.contains(values[0])) {
					posTopics.add(values[0]);
				}else if (values[3].equals("0.9") && !posTopics.contains(values[0])) {
					negTopics.add(values[0]);
				}
				
				
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		initialSentimentWord.add(posTopics);
		initialSentimentWord.add(negTopics);
		
	}

	public void prior2beta(){
		Map<String, Integer> wordmap = data.localDict.word2id;
		Iterator<String> it1 = wordmap.keySet().iterator();
		Iterator<String> it2 = wordmap.keySet().iterator();
		Iterator<String> it3 = wordmap.keySet().iterator();
			
		
		//update lambda_e
		while (it1.hasNext()){
			String key = it1.next();
			Integer value = wordmap.get(key);
			for(int e = 0; e < E; e++){
				if(initialEntityWord.get(e).contains(key)){
					for(int j = 0; j < E; j++){
						if(j != e){
							lambda_e[j][value] = 0;
						}else {
							
						}
					}
					
				}
			}
			
			
		}
		
		//update lambda_t
		while (it2.hasNext()){
			String key = it2.next();
			Integer value = wordmap.get(key);
			for(int t = 0; t < T; t++){
				if(initialTopicWord.get(t).contains(key)){
					for(int j = 0; j < T; j++){
						if(j != t){
							lambda_t[j][value] = 0;
						}
					}
					
				}
			}
			
			
		}
		
		/*
		//update lambda_s
		
		while (it3.hasNext()){
			String key = it3.next();
			Integer value = wordmap.get(key);
			for(int s = 0; s < S; s++){
				for(String word:initialSentimentWord.get(s)) {
					if(word.split("\\s+")[0].equals(key)){
						for(int j = 0; j < S; j++){
							if(j != s){
								lambda_s[j][value] = 0;
															
							} else if (word.split("\\s+")[1].equals("trust") || word.split("\\s+")[1].equals("anger")) {
								lambda_s[j][value] = 0.8;
							} else if (word.split("\\s+")[1].equals("joy") || word.split("\\s+")[1].equals("disgust")) {
								lambda_s[j][value] = 0.6;
							} else if (word.split("\\s+")[1].equals("anticip") || word.split("\\s+")[1].equals("fear")) {
								lambda_s[j][value] = 0.4;
							} else if (word.split("\\s+")[1].equals("sandness")) {
								lambda_s[j][value] = 0.2;
							} 
						}
					}
				}
					
					
				}
			
			
			
		}*/
		
/*
		while (it3.hasNext()){
			String key = it3.next();
			Integer value = wordmap.get(key);
			for(int s = 0; s < S; s++){
				for(String word:initialSentimentWord.get(s)) {
					if(word.split("\\s+")[0].equals(key)){
						for(int j = 0; j < S; j++){
							if(j != s){
								lambda_s[j][value] = 0;
															
							}  
						}
					}
				}
					
					
			
			}
			
			
		}*/
		
		
		SentiStrength sentiStrength = new SentiStrength(); 
		String ssthInitialisation[] = {"sentidata", "TopicLists/sentiment/SentiStrength_DataEnglishFeb2017/"};
		sentiStrength.initialiseAndRun(ssthInitialisation); 
		while (it3.hasNext()){
			String key = it3.next();
			Integer value = wordmap.get(key);
			String[] words = sentiStrength.computeSentimentScores(key).split("\\s+");
			int pos = Integer.parseInt(words[0]);
			int neg = Integer.parseInt(words[1]);
			//int sen = Integer.parseInt(words[2]);
			int val = 0;
			Integer sentiment = null;
			
			if (Math.abs(pos) < Math.abs(neg)) {
				sentiment = 1;
				val = neg;
			}else if(Math.abs(pos) > Math.abs(neg)) {
				sentiment = 0;
				val = pos;
			}
			
			if (sentiment != null) {
				for(int j = 0; j < S; j++){
					if( j != sentiment) {
						lambda_s[j][value] = 0.0;
					} else if (Math.abs(pos + neg) == 4) {
						lambda_s[j][value] = 0.80;
					} else if (Math.abs(pos + neg) == 3) {
						lambda_s[j][value] = 0.60;
					} else if (Math.abs(pos + neg) == 2) {
						lambda_s[j][value] = 0.40;
					} else if (Math.abs(pos + neg) == 1) {
						lambda_s[j][value] = 0.20;
					} 
					
				}
			} 
			
			
		}
		
		
		/*
		Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        while (it3.hasNext()){
			String key = it3.next();
			Integer value = wordmap.get(key);
			int mainSentiment = 0;
			int longest = 0;
			Integer sentiment = null;
	        Annotation annotation = pipeline.process(key);
			for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
	            Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
	            int sentiment_ = RNNCoreAnnotations.getPredictedClass(tree);
	            String partText = sentence.toString();
	            if (partText.length() > longest) {
	                mainSentiment = sentiment_;
	                longest = partText.length();
	            }

	        }
			if(mainSentiment == 0 || mainSentiment == 1) {
				sentiment = 1;
			} else if (mainSentiment == 3 || mainSentiment == 4) {
				sentiment  = 0;
			}
			
			if (sentiment != null) {
				for(int j = 0; j < S; j++){
					if( j != sentiment) {
						lambda_s[j][value] = 0.0;
					} else if (mainSentiment == 1 || mainSentiment == 4) {
						lambda_s[j][value] = 0.80;
					} 
					
				}
			} 
			
        }
        */
        
		
		/*
		
		while (it3.hasNext()){
			String key = it3.next();
			Integer value = wordmap.get(key);
			for(int s = 0; s < S; s++){
				for(String word:initialSentimentWord.get(s)) {
					if(word.split("\\s+")[0].equals(key)){
						for(int j = 0; j < S; j++){
							if(j != s){
								lambda_s[j][value] = 0;
															
							}  else  {
								lambda_s[j][value] = Double.parseDouble(word.split("\\s+")[1]);
							}
						}
						
					}
				}
				
			}
			
			
		}*/
		
		//lambda_e x beta_ew
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				beta_ew[e][w] = beta_ew[e][w] * lambda_e[e][w];
				
			}
			
		}
		
		//lambda_t x beta_tw
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				beta_tw[t][w] = beta_tw[t][w] * lambda_t[t][w];
				
			}
			
		}
		
		//lambda_s x beta_sw
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w] = beta_sw[s][w] * lambda_s[s][w];
				
			}
			
		}
		
	
		
		
	}

	/**
	 * init parameter for continue estimating or for later inference
	 */
	public boolean initEstimatedModel(LDACmdOption option){
		if (!init(option))
			return false;
		
			
		p = new double[E][T][S];
		
		// load model, i.e., read z and trndata
		if (!loadModel()){
			System.out.println("Fail to load word-topic assignment file of the model!\n");
			return false;
		}
		
		
		
		System.out.println("Model loaded:");
		System.out.println("\talpha_e:" + alpha_e);
		System.out.println("\talpha_t:" + alpha_t);
		System.out.println("\talpha_s:" + alpha_s);
		System.out.println("\tbeta_e:" + beta_e);
		System.out.println("\tbeta_t:" + beta_t);
		System.out.println("\tbeta_s:" + beta_s);
		System.out.println("\tM:" + M);
		System.out.println("\tV:" + V);		
		
		nwe = new int[V][E];
		for (int w = 0; w < V; w++){
			for (int e = 0; e < E; e++){
				nwe[w][e] = 0;
			}
		}
		
		nwt = new int[V][T];
		for (int w = 0; w < V; w++){
			for (int t = 0; t < T; t++){
				nwt[w][t] = 0;
			}
		}
		
		nws = new int[V][S];
		for (int w = 0; w < V; w++){
			for (int s = 0; s < S; s++){
				nws[w][s] = 0;
			}
		}
		
		ne =  new int[E];
		for (int e = 0; e < E; e++) {
			ne[e] = 0;
		}
		
		nt =  new int[T];
		for (int t = 0; t < T; t++) {
			nt[t] = 0;
		}
		
		ns =  new int[S];
		for (int s = 0; s < S; s++) {
			ns[s] = 0;
		}
		
		ndse = new int[M][S][E];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				for (int e = 0; e < E; e++) {
					ndse[m][s][e] = 0;
				}
			}			
		}
		
		ndst = new int[M][S][T];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				for (int t = 0; t < T; t++) {
					ndst[m][s][t] = 0;
				}
			}			
		}
		
		nd = new int[M];
		for (int m = 0; m < M; m++){
			nd[m] = 0;
			
		}
		
		nds =  new int[M][S];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				nds[m][s] = 0;
			}
					
			
		}
		
	    
	    for (int m = 0; m < data.M; m++){
	    	int N = data.docs[m].length;
	    	
	    	for (int n = 0; n < N; n++){
	    		int w = data.docs[m].words[n];
	    		int entity = (Integer)e[m].get(n);
	    		int topic = (Integer)t[m].get(n);
	    		int sentiment = (Integer)s[m].get(n);
	    		
	    		nwe[data.docs[m].words[n]][entity] += 1;
				nwt[data.docs[m].words[n]][topic] += 1;
				nws[data.docs[m].words[n]][sentiment] += 1;
				ne[entity] += 1;
				nt[topic] += 1;
				ns[sentiment] += 1;
				ndse[m][sentiment][entity] += 1;
				ndst[m][sentiment][topic] += 1;
				nds[m][sentiment] += 1;
				nd[m] += 1;    		
	    	}
	    	
	    }
	    
	    theta_e = new double[M][S][E];
		theta_t = new double[M][S][T];
		theta_s = new double[M][S];
		
		phi_e = new double[E][V];
		phi_t = new double[T][V];
		phi_s = new double[S][V];
		
		//initialize beta_ew with value of beta
		beta_ew = new double[E][V];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				beta_ew[e][w] = beta_e;
			}
		}
		
		//initialize beta_ew with value of beta
		beta_tw = new double[T][V];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				beta_tw[t][w] = beta_t;
			}
		}
		
		//initialize beta_ew with value of beta
		beta_sw = new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w] = beta_s;
			}
		}
		
		alpha_se = new double[S][E];
		alphaSum_se = new double[S];
		
		for(int s = 0; s < S; s++){	
				alphaSum_se[s] = 0;
				for(int e = 0; e < E; e++){
					
					alpha_se[s][e] = alpha_e;
					alphaSum_se[s] += alpha_se[s][e];
				}
			
		}
		
		alpha_st = new double[S][T];
		alphaSum_st = new double[S];
		
		
		for(int s = 0; s < S; s++){	
				alphaSum_st[s] = 0;
				for(int t = 0; t < T; t++){
					
					alpha_st[s][t] = alpha_t;
					alphaSum_st[s] += alpha_st[s][t];
				}
			
		}
		
		
				
		//Initialize lambda_e 
		lambda_e = new double[E][V];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				lambda_e[e][w] = 1;
			}
		}
		
		//Initialize lambda_t 
		lambda_t = new double[T][V];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				lambda_t[t][w] = 1;
			}
		}
		
		//Initialize lambda_s 
		lambda_s = new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				lambda_s[s][w] = 1;
			}
		}
		
		loadData();
		prior2beta();
		
		

		beta_esum = new double[E];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				beta_esum[e] += beta_ew[e][w];
			}
		}
		
		beta_tsum = new double[T];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				beta_tsum[t] += beta_tw[t][w];
			}
		}
		
		beta_ssum = new double[S];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_ssum[s] += beta_sw[s][w];
			}
		}
		
	    
		

	    
		dir = option.dir;
		savestep = option.savestep;
		
		return true;
	}
	
	/**
	 * load saved model
	 */
	public boolean loadModel(){
		if (!readOthersFile(dir + File.separator + modelName + othersSuffix))
			return false;
		
		if (!readTAssignFile(dir + File.separator + modelName + tassignSuffix))
			return false;
		
		// read dictionary
		Dictionary dict = new Dictionary();
		if (!dict.readWordMap(dir + File.separator + wordMapFile))
			return false;
			
		data.localDict = dict;
		
		return true;
	}
	
	/**
	 * read other file to get parameters
	 */
	protected boolean readOthersFile(String otherFile){
		//open file <model>.others to read:
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(otherFile));
			String line;
			while((line = reader.readLine()) != null){
				StringTokenizer tknr = new StringTokenizer(line,"= \t\r\n");
				
				int count = tknr.countTokens();
				if (count != 2)
					continue;
				
				String optstr = tknr.nextToken();
				String optval = tknr.nextToken();
				
				if (optstr.equalsIgnoreCase("alpha_e")){
					alpha_e = Double.parseDouble(optval);					
				}
				else if (optstr.equalsIgnoreCase("alpha_s")){
					alpha_s = Double.parseDouble(optval);					
				}
				else if (optstr.equalsIgnoreCase("alpha_t")){
					alpha_t = Double.parseDouble(optval);					
				}
				else if (optstr.equalsIgnoreCase("beta_e")){
					beta_e = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("beta_t")){
					beta_t = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("beta_s")){
					beta_s = Double.parseDouble(optval);
				}
				else if (optstr.equalsIgnoreCase("etopics")){
					E = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("ttopics")){
					T = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("stopics")){
					S = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("liter")){
					liter = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("nwords")){
					V = Integer.parseInt(optval);
				}
				else if (optstr.equalsIgnoreCase("ndocs")){
					M = Integer.parseInt(optval);
				}
				else {
					// any more?
				}
			}
			
			reader.close();
		}
		catch (Exception e){
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	protected boolean readTAssignFile(String tassignFile){
		try {
			int i,j;
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(tassignFile), "UTF-8"));
			
			String line;
			e = new Vector[M];	
			s = new Vector[M];
			t = new Vector[M];
			
			data = new LDADataset(M);
			data.V = V;			
			for (i = 0; i < M; i++){
				line = reader.readLine();
				StringTokenizer tknr = new StringTokenizer(line, " \t\r\n");
				
				int length = tknr.countTokens();
				
				Vector<Integer> words = new Vector<Integer>();
				Vector<Integer> entities = new Vector<Integer>();
				Vector<Integer> topics = new Vector<Integer>();
				Vector<Integer> sentiment = new Vector<Integer>();
				
				for (j = 0; j < length; j++){
					String token = tknr.nextToken();
					
					StringTokenizer tknr2 = new StringTokenizer(token, ":");
					if (tknr2.countTokens() != 4){
						System.out.println("Invalid word-topic assignment line\n");
						return false;
					}
					
					words.add(Integer.parseInt(tknr2.nextToken()));
					entities.add(Integer.parseInt(tknr2.nextToken()));
					topics.add(Integer.parseInt(tknr2.nextToken()));
					sentiment.add(Integer.parseInt(tknr2.nextToken()));
				}//end for each topic assignment
				
				//allocate and add new document to the corpus
				Document doc = new Document(words);
				data.setDoc(doc, i);
				
				//assign values for e
				e[i] = new Vector<Integer>();
				for (j = 0; j < entities.size(); j++){
					e[i].add(entities.get(j));
				}
				
				//assign values for t
				t[i] = new Vector<Integer>();
				for (j = 0; j < topics.size(); j++){
					t[i].add(topics.get(j));
				}
				
				//assign values for s
				s[i] = new Vector<Integer>();
				for (j = 0; j < sentiment.size(); j++){
					s[i].add(sentiment.get(j));
				}
			}//end for each doc
			
			reader.close();
		}
		catch (Exception e){
			System.out.println("Error while loading model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model
	 */
	public boolean saveModel(String modelName){
		if (!saveModelTAssign(dir + File.separator + modelName + tassignSuffix)){
			return false;
		}
		
		if (!saveModelOthers(dir + File.separator + modelName + othersSuffix)){			
			return false;
		}
		
		if (!saveModelThetaE(dir + File.separator + modelName + "_E" +thetaSuffix)){
			return false;
		}
		
		if (!saveModelThetaT(dir + File.separator + modelName + "_T" +thetaSuffix)){
			return false;
		}
		
		if (!saveModelThetaS(dir + File.separator + modelName + "_S" +thetaSuffix)){
			return false;
		}
		
		if (!saveModelPhiE(dir + File.separator + modelName + "_E" + phiSuffix)){
			return false;
		}
		
		if (!saveModelPhiT(dir + File.separator + modelName + "_T" + phiSuffix)){
			return false;
		}
		
		if (!saveModelPhiS(dir + File.separator + modelName + "_S" + phiSuffix)){
			return false;
		}
		
		if (twords > 0){
			if (!saveModelEwords(dir + File.separator + modelName + "_E"+ twordsSuffix))
				return false;
		}
		
		if (twords > 0){
			if (!saveModelTwords(dir + File.separator + modelName + "_T"+ twordsSuffix))
				return false;
		}
		
		if (twords > 0){
			if (!saveModelSwords(dir + File.separator + modelName + "_S"+ twordsSuffix))
				return false;
		}
		return true;
	}
	
	/**
	 * Save word-topic assignments for this model
	 */
	public boolean saveModelTAssign(String filename){
		int i, j;
		
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			//write docs with topic assignments for words
			for (i = 0; i < data.M; i++){
				for (j = 0; j < data.docs[i].length; ++j){
					writer.write(data.docs[i].words[j] + ":" + e[i].get(j) + ":" + t[i].get(j) + ":" + s[i].get(j) + " ");					
				}
				writer.write("\n");
			}
				
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving model tassign: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save other information of this model
	 */
	public boolean saveModelOthers(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			writer.write("alpha_e=" + alpha_e + "\n");
			writer.write("alpha_t=" + alpha_t + "\n");
			writer.write("alpha_s=" + alpha_s + "\n");
			writer.write("beta_e=" + beta_e + "\n");
			writer.write("beta_t=" + beta_t + "\n");
			writer.write("beta_s=" + beta_s + "\n");
			writer.write("etopics=" + E + "\n");
			writer.write("ttopics=" + T + "\n");
			writer.write("stopics=" + S + "\n");
			writer.write("ndocs=" + M + "\n");
			writer.write("nwords=" + V + "\n");
			writer.write("liters=" + liter + "\n");
			
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model others:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save theta (Entity distribution) for this model
	 */
	public boolean saveModelThetaE(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i < M; i++){
				for (int s = 0; s < S; s++){
					for (int e = 0; e < E; e++){
						writer.write(theta_e[i][s][e] + " ");
					}
				}
				
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save theta (Topic distribution) for this model
	 */
	public boolean saveModelThetaT(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i < M; i++){
				for (int s = 0; s < S; s++){
					for (int t = 0; t < T; t++){
						writer.write(theta_t[i][s][t] + " ");
					}
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save theta (topic distribution) for this model
	 */
	public boolean saveModelThetaS(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i < M; i++){
				
				for(int s = 0; s < S; s++) {
					writer.write(theta_s[i][s] + " ");
				}
						
			writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save word-entity distribution
	 */
	
	public boolean saveModelPhiE(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int e = 0; e < E; e++){
				for (int j = 0; j < V; j++){
					writer.write(phi_e[e][j] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save word-topic distribution
	 */
	
	public boolean saveModelPhiT(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int t = 0; t < T; t++){
				for (int j = 0; j < V; j++){
					writer.write(phi_t[t][j] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save word-sentiment distribution
	 */
	
	public boolean saveModelPhiS(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			
			for (int s = 0; s < S; s++){
				for (int j = 0; j < V; j++){
					writer.write(phi_s[s][j] + " ");
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch (Exception e){
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model the most likely words for each entity
	 */
	public boolean saveModelEwords(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int e = 0; e < E; e++){
				List<Pair> wordsProbsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, phi_e[e][w], false);
					
					wordsProbsList.add(p);
				}//end foreach word
				
				//print topic				
				writer.write("Entity " + e + "th:\n");
				Collections.sort(wordsProbsList);
				
				for (int i = 0; i < twords; i++){
					if (data.localDict.contains((Integer)wordsProbsList.get(i).first)){
						String word = data.localDict.getWord((Integer)wordsProbsList.get(i).first);
						
						writer.write("\t" + word + " " + wordsProbsList.get(i).second + "\n");
					}
				}
			} //end foreach topic			
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model the most likely words for each entity
	 */
	public boolean saveModelTwords(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int t = 0; t < T; t++){
				List<Pair> wordsProbsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, phi_t[t][w], false);
					
					wordsProbsList.add(p);
				}//end foreach word
				
				//print topic				
				writer.write("Topic " + t + "th:\n");
				Collections.sort(wordsProbsList);
				
				for (int i = 0; i < twords; i++){
					if (data.localDict.contains((Integer)wordsProbsList.get(i).first)){
						String word = data.localDict.getWord((Integer)wordsProbsList.get(i).first);
						
						writer.write("\t" + word + " " + wordsProbsList.get(i).second + "\n");
					}
				}
			} //end foreach topic			
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	/**
	 * Save model the most likely words for each entity
	 */
	public boolean saveModelSwords(String filename){
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream(filename), "UTF-8"));
			
			if (twords > V){
				twords = V;
			}
			
			for (int s = 0; s < S; s++){
				List<Pair> wordsProbsList = new ArrayList<Pair>(); 
				for (int w = 0; w < V; w++){
					Pair p = new Pair(w, phi_s[s][w], false);
					
					wordsProbsList.add(p);
				}//end foreach word
				
				//print topic				
				writer.write("Sentiment " + s + "th:\n");
				Collections.sort(wordsProbsList);
				
				for (int i = 0; i < twords; i++){
					if (data.localDict.contains((Integer)wordsProbsList.get(i).first)){
						String word = data.localDict.getWord((Integer)wordsProbsList.get(i).first);
						
						writer.write("\t" + word + " " + wordsProbsList.get(i).second + "\n");
					}
				}
			} //end foreach topic			
						
			writer.close();
		}
		catch(Exception e){
			System.out.println("Error while saving model twords: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * Init parameters for inference
	 * reading new dataset from file
	 */
	public boolean initNewModel(LDACmdOption option, Model trnModel){
		if (!init(option))
			return false;
		
		LDADataset dataset = LDADataset.readDataSet(dir + File.separator + dfile, trnModel.data.localDict);
		if (dataset == null){
			System.out.println("Fail to read dataset!\n");
			return false;
		}
		
		return initNewModel(option, dataset , trnModel);
	}
	

	/**
	 * Init parameters for inference
	 * @param newData DataSet for which we do inference
	 */
	public boolean initNewModel(LDACmdOption option, LDADataset newData, Model trnModel){
		if (!init(option))
			return false;
		
				
		E = trnModel.E;
		S = trnModel.S;
		T = trnModel.T;
		alpha_e = trnModel.alpha_e;
		alpha_s = trnModel.alpha_s;
		alpha_t = trnModel.alpha_t;
		beta_e = trnModel.beta_e;
		beta_s = trnModel.beta_s;
		beta_t = trnModel.beta_t;
		
		p = new double[E][T][S];
				
		data = newData;
		
		//+ allocate memory and assign values for variables		
		M = data.M;
		V = data.V;
		dir = option.dir;
		savestep = option.savestep;
		System.out.println("M:" + M);
		System.out.println("V:" + V);
		
		// K: from command line or default value
	    // alpha, beta: from command line or default values
	    // niters, savestep: from command line or default values
		nwe = new int[V][E];
		for (int w = 0; w < V; w++){
			for (int e = 0; e < E; e++){
				nwe[w][e] = 0;
			}
		}
		
		nwt = new int[V][T];
		for (int w = 0; w < V; w++){
			for (int t = 0; t < T; t++){
				nwt[w][t] = 0;
			}
		}
		
		nws = new int[V][S];
		for (int w = 0; w < V; w++){
			for (int s = 0; s < S; s++){
				nws[w][s] = 0;
			}
		}
		
		ne =  new int[E];
		for (int e = 0; e < E; e++) {
			ne[e] = 0;
		}
		
		nt =  new int[T];
		for (int t = 0; t < T; t++) {
			nt[t] = 0;
		}
		
		ns =  new int[S];
		for (int s = 0; s < S; s++) {
			ns[s] = 0;
		}
		
		ndse = new int[M][S][E];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				for (int e = 0; e < E; e++) {
					ndse[m][s][e] = 0;
				}
			}			
		}
		
		ndst = new int[M][S][T];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				for (int t = 0; t < T; t++) {
					ndst[m][s][t] = 0;
				}
			}			
		}
		
		nd = new int[M];
		for (int m = 0; m < M; m++){
			nd[m] = 0;
			
		}
		
		nds =  new int[M][S];
		for (int m = 0; m < M; m++){
			for (int s = 0; s < S; s++) {
				nds[m][s] = 0;
			}
					
			
		}
		
		
		loadData();
		
		e = new Vector[M];
		t = new Vector[M];
		s = new Vector[M];
		
		SentiStrength sentiStrength = new SentiStrength(); 
		String ssthInitialisation[] = {"sentidata", "TopicLists/sentiment/SentiStrength_DataEnglishFeb2017/", "binary"};
		sentiStrength.initialiseAndRun(ssthInitialisation); 
		
		for (int m = 0; m < data.M; m++){
			int N = data.docs[m].length;
			e[m] = new Vector<Integer>();
			t[m] = new Vector<Integer>();
			s[m] = new Vector<Integer>();
			
			StringTokenizer tknr = new StringTokenizer(data.docs[m].rawStr, " \t\r\n");
			
			//initilize for e, t and s
			for (int n = 0; n < N; n++){
				String token=tknr.nextToken();
				Integer entity = null;
				Integer topic = null;
				Integer sentiment = null;
				
				int i = 0;
				for(List<String> entityList:initialEntityWord){
					
					for(String entityWord:entityList){
						if(token.equals(entityWord.toLowerCase())){
							entity=i;
							break;
						}
					}
					i++;
					if(entity!=null){
						break;
					}
				}
				
				i = 0;
				for(List<String> topicList:initialTopicWord){
					
					for(String topicWord:topicList){
						if(token.equals(topicWord.toLowerCase())){
							topic=i;
							break;
						}
					}
					i++;
					if(topic!=null){
						break;
					}
				}
				/*
				i=0;
				for(List<String> sentimentList:initialSentimentWord){
					
					for(String sentimentWord:sentimentList){
						if(token.equals(sentimentWord.toLowerCase())){
							sentiment=i;
							break;
						}
					}
					i++;
					if(sentiment!=null){
						break;
					}
				}*/
				
				String[] words = sentiStrength.computeSentimentScores(token).split("\\s+");
				int pos = Integer.parseInt(words[0]);
				int neg = Integer.parseInt(words[1]);
				//int sen = Integer.parseInt(words[2]);
				
				if (Math.abs(pos) < Math.abs(neg)) {
					sentiment = 1;
				}else if(Math.abs(pos) > Math.abs(neg)) {
					sentiment = 0;
				}
				
				
				if(entity == null){
					entity = (int)Math.floor(Math.random() * E);
				}
				if(topic == null){
					topic = (int)Math.floor(Math.random() * T);
				}
				if(sentiment == null){
					sentiment = (int)Math.floor(Math.random() * S);
				}
				
				e[m].add(entity);
				t[m].add(topic);
				s[m].add(sentiment);
				
				nwe[data.docs[m].words[n]][entity] += 1;
				nwt[data.docs[m].words[n]][topic] += 1;
				nws[data.docs[m].words[n]][sentiment] += 1;
				ne[entity] += 1;
				nt[topic] += 1;
				ns[sentiment] += 1;
				ndse[m][sentiment][entity] += 1;
				ndst[m][sentiment][topic] += 1;
				nds[m][sentiment] += 1;
				nd[m] += 1;
			}
			
			
			
			
		}
		
		
		theta_e = new double[M][S][E];
		theta_t = new double[M][S][T];
		theta_s = new double[M][S];
		
		phi_e = new double[E][V];
		phi_t = new double[T][V];
		phi_s = new double[S][V];
		
		//initialize beta_ew with value of beta
		beta_ew=new double[E][V];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				beta_ew[e][w]=beta_e;
			}
		}
		
		//initialize beta_ew with value of beta
		beta_tw=new double[T][V];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				beta_tw[t][w]=beta_t;
			}
		}
		
		//initialize beta_ew with value of beta
		beta_sw=new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_sw[s][w]=beta_s;
			}
		}
		
		alpha_se = new double[S][E];
		alphaSum_se = new double[S];
		
		for(int s = 0; s < S; s++){	
				alphaSum_se[s] = 0;
				for(int e = 0; e < E; e++){
					
					alpha_se[s][e] = alpha_e;
					alphaSum_se[s] += alpha_se[s][e];
				}
			
		}
		
		alpha_st = new double[S][T];
		alphaSum_st = new double[S];
		
		
		for(int s = 0; s < S; s++){	
				alphaSum_st[s] = 0;
				for(int t = 0; t < T; t++){
					
					alpha_st[s][t] = alpha_t;
					alphaSum_st[s] += alpha_st[s][t];
				}
			
		}
		
		
				
		//Initialize lambda_e 
		lambda_e=new double[E][V];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				lambda_e[e][w] = 1;
			}
		}
		
		//Initialize lambda_t 
		lambda_t=new double[T][V];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				lambda_t[t][w] = 1;
			}
		}
		
		//Initialize lambda_s 
		lambda_s=new double[S][V];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				lambda_s[s][w] = 1;
			}
		}
		
		
		prior2beta();
		
		

		beta_esum=new double[E];
		for(int e = 0; e < E; e++){
			for(int w = 0; w < V; w++){
				beta_esum[e] += beta_ew[e][w];
			}
		}
		
		beta_tsum=new double[T];
		for(int t = 0; t < T; t++){
			for(int w = 0; w < V; w++){
				beta_tsum[t] += beta_tw[t][w];
			}
		}
		
		beta_ssum=new double[S];
		for(int s = 0; s < S; s++){
			for(int w = 0; w < V; w++){
				beta_ssum[s] += beta_sw[s][w];
			}
		}		
		
		
		return true;
	}
	
	
}
