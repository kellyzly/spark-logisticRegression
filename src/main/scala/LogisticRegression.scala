
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler

object  LogisticRegression{

    /**
      * load data from banking_marketing_data.csv and add new column "jobIndex","jobVec"
      * "maritalIndex","maritalVec", "defaultIndex", "defaultVec", "housingIndex", "housingVec",
      * "poutcomeIndex",""poutcomeVec,"loadIndex", "loadVec"
      */
    def buildLogisticRegression :Unit={
        val spark = SparkSession.builder().appName("Logistic_Prediction").master("local").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        import spark.implicits._
        val bank_Marketing_Data = spark.read.option("header",true).option("inferSchema","true").csv("C:\\Users\\zly\\Documents\\eBay\\scala_spark_study\\spark-app\\src\\main\\resources\\bank_marketing_data.csv")
        bank_Marketing_Data.show(5)
        //change column "age","duration","previous" from type Int to Double ,
        val selected_Data =bank_Marketing_Data.select("age","job",
            "marital","default","housing","loan","duration","previous",
            "poutcome","empvarrate","y").withColumn("age",bank_Marketing_Data("age").cast(DoubleType)).withColumn("duration",
            bank_Marketing_Data("duration").cast(DoubleType)).withColumn("previous",bank_Marketing_Data("previous").cast(DoubleType))
        //change to matrix
        //add new column jonIndex and jobVec
        val indexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
        val indexed = indexer.fit(selected_Data).transform(selected_Data)
        indexed.printSchema()
        indexed.show
        val encoder = new OneHotEncoder().setDropLast(false).setInputCol("jobIndex").setOutputCol("jobVec")
        val encoded = encoder.transform(indexed)
        //add new column maritalIndex and maritalVec
        val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
        //notice here we use the "encoded" as input data not "selected_Data" as in "selected_Data" there is no
        //column "jobVec"
        val maritalIndexed = maritalIndexer.fit(encoded).transform(encoded)
        val maritalEncoder = new OneHotEncoder().setDropLast(false).setInputCol("maritalIndex").setOutputCol("maritalVec")
        val maritalEncoded = maritalEncoder.transform(maritalIndexed)

        val defaultIndexer = new StringIndexer().setInputCol("default").setOutputCol("defaultIndex")
        val defaultIndexed = defaultIndexer.fit(maritalEncoded).transform(maritalEncoded)
        val defaultEncoder = new OneHotEncoder().setDropLast(false).setInputCol("defaultIndex").setOutputCol("defaultVec")
        val defaultEncoded = defaultEncoder.transform(defaultIndexed)

        val housingIndexer = new StringIndexer().setInputCol("housing").setOutputCol("housingIndex")
        val housingIndexed = housingIndexer.fit(defaultEncoded).transform(defaultEncoded)
        val housingEncoder = new OneHotEncoder().setDropLast(false).setInputCol("housingIndex").setOutputCol("housingVec")
        val housingEncoded = housingEncoder.transform(housingIndexed)

        val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")
        val poutcomeIndexed = poutcomeIndexer.fit(housingEncoded).transform(housingEncoded)
        val poutcomeEncoder = new OneHotEncoder().setDropLast(false).setInputCol("poutcomeIndex").setOutputCol("poutcomeVec")
        val poutcomeEncoded = poutcomeEncoder.transform(poutcomeIndexed)

        val loanIndexer = new StringIndexer().setInputCol("loan").setOutputCol("loanIndex")
        val loanIndexed = loanIndexer.fit(poutcomeEncoded).transform(poutcomeEncoded)
        val loanEncoder = new OneHotEncoder().setDropLast(false).setInputCol("loanIndex").setOutputCol("loanVec")
        val loanEncoded = loanEncoder.transform(loanIndexed)

        loanEncoded.show()
        loanEncoded.printSchema()

        val vectorAssembler = new VectorAssembler().setInputCols(Array("jobVec","maritalVec","defaultVec","housingVec",
            "poutcomeVec","loanVec","age","duration","previous","empvarrate")).setOutputCol("features")
        val indexerY  = new StringIndexer().setInputCol("y").setOutputCol("label")
        val transformers = Array(indexer, encoder, maritalIndexer, maritalEncoder, defaultIndexer, defaultEncoder,housingIndexer,
            housingEncoder,poutcomeIndexer,poutcomeEncoder,loanIndexer, loanEncoder,vectorAssembler,indexerY)

        // split original data into 8:2  80%：training 20%：test
        val splits = selected_Data.randomSplit(Array(0.8,0.2))
        val training = splits(0).cache()
        val test = splits(1).cache()

        val lr = new LogisticRegression()
        //将算法数组和逻辑回归算法合并，传入pipeline对象的stages中，然后作用于训练数据，训练模型
        var model = new Pipeline().setStages(transformers:+lr).fit(training)
        // 将上一步的训练模型作用于测试数据，返回测试结果
        var result = model.transform(test)
        // 显示测试结果集中的真实值，预测值，原始值，百分比字段
        result.select("label","prediction","rawPrediction","probability").show(10,false)
        // 创建二分类算法评估器，对测试结果进行评估
        val evaluator = new BinaryClassificationEvaluator()
        var aucTraining = evaluator.evaluate(result)
        //  对于20%的测试集的精准度为0.92 说明预测准确度高
        println("aucTraining = "+aucTraining)

    }



    def main(args: Array[String]): Unit = {
     buildLogisticRegression
//        only showing top 20 rows
//
//        root
//        |-- age: double (nullable = true)
//        |-- job: string (nullable = true)
//        |-- marital: string (nullable = true)
//        |-- default: string (nullable = true)
//        |-- housing: string (nullable = true)
//        |-- loan: string (nullable = true)
//        |-- duration: double (nullable = true)
//        |-- previous: double (nullable = true)
//        |-- poutcome: string (nullable = true)
//        |-- empvarrate: double (nullable = true)
//        |-- y: string (nullable = true)
//        |-- jobIndex: double (nullable = false)
//        |-- jobVec: vector (nullable = true)
//        |-- maritalIndex: double (nullable = false)
//        |-- maritalVec: vector (nullable = true)
//        |-- defaultIndex: double (nullable = false)
//        |-- defaultVec: vector (nullable = true)
//        |-- housingIndex: double (nullable = false)
//        |-- housingVec: vector (nullable = true)
//        |-- poutcomeIndex: double (nullable = false)
//        |-- poutcomeVec: vector (nullable = true)
//        |-- loanIndex: double (nullable = false)
//        |-- loanVec: vector (nullable = true)
//
//        19/04/01 16:54:53 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
//        19/04/01 16:54:53 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
//        19/04/01 16:54:56 WARN Utils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.debug.maxToStringFields' in SparkEnv.conf.
//          +-----+----------+------------------------------------------+----------------------------------------+
//        |label|prediction|rawPrediction                             |probability                             |
//        +-----+----------+------------------------------------------+----------------------------------------+
//        |0.0  |1.0       |[-0.886162936645015,0.886162936645015]    |[0.29190229682492363,0.7080977031750764]|
//          |0.0  |0.0       |[1.734423908301129,-1.734423908301129]    |[0.8499774123460015,0.15002258765399853]|
//          |1.0  |1.0       |[-1.1167006026223496,1.1167006026223496]  |[0.24662380062571557,0.7533761993742845]|
//          |0.0  |1.0       |[-0.37455715498427544,0.37455715498427544]|[0.4074403129316124,0.5925596870683877] |
//          |0.0  |0.0       |[1.6141579965950528,-1.6141579965950528]  |[0.8339878695407037,0.1660121304592962] |
//          |1.0  |1.0       |[-2.2248600736109183,2.2248600736109183]  |[0.0975401535175688,0.9024598464824312] |
//          |0.0  |0.0       |[1.6461732794209425,-1.6461732794209425]  |[0.8383731872538406,0.1616268127461593] |
//          |1.0  |0.0       |[1.0206817897541687,-1.0206817897541687]  |[0.7351053825683843,0.26489461743161563]|
//          |1.0  |1.0       |[-1.2224665532760701,1.2224665532760701]  |[0.22750267432215562,0.7724973256778445]|
//          |1.0  |0.0       |[0.3371089474754805,-0.3371089474754805]  |[0.5834880812277123,0.4165119187722877] |
//          +-----+----------+------------------------------------------+----------------------------------------+
//        only showing top 10 rows
//
//        aucTraining = 0.9219319540743842only showing top 20 rows


}

}