using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    class Program
    {
        class SentimentData
        {
            [LoadColumn(0)]
            public string SentimentText;

            [LoadColumn(1), ColumnName("Label")]
            public bool Sentiment;
        }

        class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction;
        }
        static void Main(string[] args)
        {

            // Step 2: Create a ML.NET environment
            var mlContext = new MLContext();

            // Step 3: Load data
            var data = mlContext.Data.LoadFromTextFile<SentimentData>("data/sentiment_data.csv", hasHeader: false);

            // Step 4: Build a data processing pipeline
            // Step 4: Build a data processing pipeline
var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));


            // Step 5: Train the model
            var model = pipeline.Fit(data);

            // Step 6: Make predictions
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            var prediction = predictionEngine.Predict(new SentimentData { SentimentText = "I love ML.NET!" });

            Console.WriteLine($"Predicted sentiment: {(prediction.Prediction ? "Positive" : "Negative")}");
        }
    }
}
