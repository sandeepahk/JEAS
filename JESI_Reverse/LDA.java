package jeas_reverse;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;



public class LDA {
	
	public static void main(String[] args) {
		for(int i = 1; i <= 10; i++) {
			LDACmdOption option = new LDACmdOption();
			CmdLineParser parser = new CmdLineParser(option);
			args = new String[]{"-est", "-dfile", "train_new.dat", "-dir", "models/JEAS-Reverse/"+i, "-etopics", "18", "-stopics", "2", "-ttopics", "16", "-twords", "20", "-alpha_e", "0.0005", "-beta_e", "1.5", "-alpha_t", "0.0006", "-beta_t", "1", "-alpha_s", "0.225", "-beta_s", "1"};
			//args = new String[] {"-inf", "-dir", "models/JEAS-Reverse/"+i, "-model", "model-final", "-niters", "1000", "-twords", "20", "-dfile", "test_new.dat"};
			
			try {
				if (args.length == 0){
					showHelp(parser);
					return;
				}
				
				parser.parseArgument(args);
				
				if (option.est || option.estc){
					Estimator estimator = new Estimator();
					estimator.init(option);
					estimator.estimate();
				}
				else if (option.inf){
					Inferencer inferencer = new Inferencer();
					inferencer.init(option);
					
					Model newModel = inferencer.inference();
				
					
				}
			}
			catch (CmdLineException cle){
				System.out.println("Command line error: " + cle.getMessage());
				showHelp(parser);
				return;
			}
			catch (Exception e){
				System.out.println("Error in main: " + e.getMessage());
				e.printStackTrace();
				return;
			}

		}
		
	}
	
	public static void showHelp(CmdLineParser parser){
		System.out.println("LDA [options ...] [arguments...]");
		parser.printUsage(System.out);
	}
	

}
