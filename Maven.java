<projectxmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https
://maven.apache.org/xsd/maven-4.0.0.xsd">
<modelVersion>4.0.0</modelVersion>
<groupId>com</groupId>

<artifactId>org.ml</artifactId>
<version>0.0.1-SNAPSHOT</version>
<dependencies>
<dependency>
<groupId>nz.ac.waikato.cms.weka</groupId>
<artifactId>weka-stable</artifactId>
<version>3.8.0</version>
</dependency>
<dependency>
<groupId>tech.tablesaw</groupId>
<artifactId>tablesaw-core</artifactId>
<version>0.38.1</version>
</dependency><dependency>
<groupId>tech.tablesaw</groupId>
<artifactId>tablesaw-jsplot</artifactId>

<version>0.38.1</version>
</dependency><!-- Thanks for using https:/jar-download.com --></dependencies>
<properties>
</properties>
</project>

<maven.compiler.source>1.8</maven.compiler.source>
<maven.compiler.target>1.8</maven.compiler.target>
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
