package com.hyunje.jo.spark;


import org.apache.commons.cli.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

/**
 * Spark를 이용해서 Wordcount를 수행하는 프로그램.
 *
 *
 <pre>
    Usage:
        $ spark-submit --class com.hyunje.jo.spark.WordCount --master yarn-cluster spark-example.jar -i [HDFS input Path] -o [HDFS output path]
 </pre>
 *
 * @author  hyunje
 * @since   2014.10.10
 */
public class WordCount {
    public static void main(String[] args) throws ParseException {
        String INPUT_PATH="", OUTPUT_PATH="";

        //Commandline Parsing
        Options options = new Options();
        options.addOption("i","input",true,"input path(HDFS)");
        options.addOption("o","output",true,"output path(HDFS)");

        CommandLineParser parser = new BasicParser();
        CommandLine cmd = parser.parse(options,args);
        if(cmd.hasOption("i")){
            INPUT_PATH = cmd.getOptionValue("i");
        } else{
            System.err.println("Input path is invalid");
        }

        if(cmd.hasOption("o")){
            OUTPUT_PATH = cmd.getOptionValue("o");
        } else{
            System.err.println("Output path is invalid");
        }

        //Create spark context
        SparkConf conf = new SparkConf().setAppName("Spark Word-count").setMaster("yarn-cluster");
        JavaSparkContext context = new JavaSparkContext(conf);

        //Split using space
        JavaRDD<String> lines = context.textFile(INPUT_PATH);
        JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterable<String> call(String s) throws Exception {
                return Arrays.asList(s.split(" "));
            }
        });

        //Generate count of word.
        JavaPairRDD<String, Integer> onesOfWord = words.mapToPair(new PairFunction<String, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String s) throws Exception {
                return new Tuple2<String, Integer>(s, 1);
            }
        });

        //Combine the count.
        JavaPairRDD<String, Integer> wordCount = onesOfWord.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer integer, Integer integer2) throws Exception {
                return integer + integer2;
            }
        });

        //Save as text file.
        wordCount.saveAsTextFile(OUTPUT_PATH);

        context.stop();
    }
}
