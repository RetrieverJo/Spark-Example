package com.hyunje.jo.spark;

import org.apache.commons.cli.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;

/**
 * Spark의 MLlib 에 존재하는 ALS 알고리즘을 이용하여 추천을 수행하는 프로그램.
 * MovieLens 데이터셋에 대해서
 * <p/>
 * <pre>
    Usage:
        $ spark-submit --class com.hyunje.jo.spark.CollaborativeFiltering --master yarn-cluster spark-example.jar -i [HDFS input Path] -o [HDFS output path]
 * </pre>
 *
 * @author hyunje
 * @since 14. 11. 3.
 */
public class CollaborativeFiltering {
    public static void main(String[] args) throws ParseException {
        String INPUT_PATH = "", OUTPUT_PATH = "";

        //Commandline Parsing
        Options options = new Options();
        options.addOption("i", "input", true, "input path(HDFS)");
        options.addOption("o", "output", true, "output path(HDFS)");

        CommandLineParser parser = new BasicParser();
        CommandLine cmd = parser.parse(options, args);
        if (cmd.hasOption("i")) {
            INPUT_PATH = cmd.getOptionValue("i");
        } else {
            System.err.println("Input path is invalid");
        }

        if (cmd.hasOption("o")) {
            OUTPUT_PATH = cmd.getOptionValue("o");
        } else {
            System.err.println("Output path is invalid");
        }

        SparkConf conf = new SparkConf().setAppName("Spark-recommendation").setMaster("yarn-cluster");
        JavaSparkContext context = new JavaSparkContext(conf);

        JavaRDD<String> data = context.textFile(INPUT_PATH);
        JavaRDD<Rating> ratings = data.map(
                new Function<String, Rating>() {
                    public Rating call(String s) {
                        String[] sarray = s.split("::");
                        return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
                                Double.parseDouble(sarray[2]));
                    }
                }
        );

        // Build the recommendation model using ALS
        int rank = 10;
        int numIterations = 20;
        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

        // Evaluate the model on rating data
        JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );
        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(
                                        new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
                            }
                        }
                ));

        //<<Integer,Integer>,Double> to <Integer,<Integer,Double>>
        JavaPairRDD<Integer, Tuple2<Integer, Double>> userPredictions = JavaPairRDD.fromJavaRDD(predictions.map(
                new Function<Tuple2<Tuple2<Integer, Integer>, Double>, Tuple2<Integer, Tuple2<Integer, Double>>>() {
                    @Override
                    public Tuple2<Integer, Tuple2<Integer, Double>> call(Tuple2<Tuple2<Integer, Integer>, Double> v1) throws Exception {
                        return new Tuple2<Integer, Tuple2<Integer, Double>>(v1._1()._1(), new Tuple2<Integer, Double>(v1._1()._2(), v1._2()));
                    }
                }
        ));

        //Sort by key & Save
        userPredictions.sortByKey(true).saveAsTextFile(OUTPUT_PATH);
        context.stop();
    }
}