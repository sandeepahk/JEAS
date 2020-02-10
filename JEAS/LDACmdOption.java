package jeas;

import org.kohsuke.args4j.Option;

public class LDACmdOption {
	@Option(name="-est", usage="Specify whether we want to estimate model from scratch")
	public boolean est = false;
	
	@Option(name="-estc", usage="Specify whether we want to continue the last estimation")
	public boolean estc = false;
	
	@Option(name="-inf", usage="Specify whether we want to do inference")
	public boolean inf = true;
	
	@Option(name="-dir", usage="Specify directory")
	public String dir = "";
	
	@Option(name="-dfile", usage="Specify data file")
	public String dfile = "";
	
	@Option(name="-model", usage="Specify the model name")
	public String modelName = "";
	
	@Option(name="-alpha_e", usage="Specify alpha_e")
	public double alpha_e = -1.0;
	
	@Option(name="-alpha_s", usage="Specify alpha_s")
	public double alpha_s = -1.0;
	
	@Option(name="-alpha_t", usage="Specify alpha_t")
	public double alpha_t = -1.0;
	
	@Option(name="-beta_e", usage="Specify beta_e")
	public double beta_e = -1.0;
	
	@Option(name="-beta_s", usage="Specify beta_s")
	public double beta_s = -1.0;
	
	@Option(name="-beta_t", usage="Specify beta_t")
	public double beta_t = -1.0;
	
	@Option(name="-etopics", usage="Specify the number of entities")
	public int E = 100;
	
	@Option(name="-stopics", usage="Specify the number of sentiment")
	public int S = 100;
	
	@Option(name="-ttopics", usage="Specify the number of topics")
	public int T = 100;
	
	@Option(name="-niters", usage="Specify the number of iterations")
	public int niters = 1000;
	
	@Option(name="-savestep", usage="Specify the number of steps to save the model since the last save")
	public int savestep = 100;
	
	@Option(name="-twords", usage="Specify the number of most likely words to be printed for each topic")
	public int twords = 100;
	
	@Option(name="-withrawdata", usage="Specify whether we include raw data in the input")
	public boolean withrawdata = false;
	
	@Option(name="-wordmap", usage="Specify the wordmap file")
	public String wordMapFileName = "wordmap.txt";
	
	@Option(name="-updateParaSteps", usage="Specify the number of steps to update parameters")
	public int updateParaSteps = 40;

}
