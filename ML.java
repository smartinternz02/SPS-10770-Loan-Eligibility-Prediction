package org.ml;
import java.io.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.RandomForest;  
import weka.core.Instances;
public class Charan {

	private static String getCurrentDirectoryPath() {
		String path=System.getProperty("user.dir")+"\\src\\main\\java\\org\\ml";
		//System.out.println(path);
		return path.replace("\\","\\\\");
	}
	public static String trainDataPath="";
	public static String testDataPath="";
	public static void main(String args[]) throws Exception{
		String trainFileName="\\\\cleaned_data_charan.csv.arff";
		String testFileName="\\\\test_charan.arff";
		trainDataPath=getCurrentDirectoryPath()+trainFileName;
		//System.out.println(trainDataPath);
		testDataPath=getCurrentDirectoryPath()+testFileName;
		BufferedReader br=new BufferedReader(new FileReader(trainDataPath));
		Instances trainData=new Instances(br);
		BufferedReader brTest=new BufferedReader(new FileReader(testDataPath));
		Instances testData=new Instances(brTest);
		testData.setClassIndex(testData.numAttributes()-1);
		trainData.setClassIndex(trainData.numAttributes()-1);
		br.close();
		logisticRegression(testData,trainData);
	}
	private static void logisticRegression(Instances testData,Instances trainData) throws Exception{
		System.out.println("\n\nLOGISTIC REGRESSION\n");
		Classifier classifier = new weka.classifiers.functions.Logistic();
		classifier.buildClassifier(trainData);
		Evaluation eval = new Evaluation(trainData);
		eval.evaluateModel(classifier,trainData);
		System.out.println(eval.toSummaryString());
		System.out.println("No.\tTrue\tPredicted");
		for(int i=0;i<testData.numInstances();i++) {
			String trueLabel;
			trueLabel=testData.instance(i).toString(testData.classIndex());
			double predIndex=classifier.classifyInstance(testData.instance(i));
			String predLabel;
			System.out.println((i+1)+"\t"+trueLabel+"\t"+predIndex);
		}   
	}

}