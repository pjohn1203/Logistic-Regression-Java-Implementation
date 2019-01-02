//package cmps142_hw4;


import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.lang.Math;

public class LogisticRegression_withRegularization {

        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the regularization coefficient */
        private double lambda = 0.001;

        /** the number of iterations */
        private int ITERATIONS = 200;

        /** TODO: Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        /** DONE **/
        public LogisticRegression_withRegularization(int n) { // n is the number of weights to be learned
          weights = new double[n];
        }

        /** TODO: Implement the function that returns the L2 norm of the weight vector **/
        /** DONE **/
        private double weightsL2Norm(){
            /** weights array. **/
            /** square each value in the vector **/
            /** sum them together */
            /** sqrt the the sum **/

            double sum = 0.0;
            for(int i=0; i < weights.length; i++){
              sum += Math.pow(weights[i] , 2);
            }
            double normVal = Math.sqrt(sum);

            return normVal;
        }

        /** TODO: Implement the sigmoid function **/
        /** DONE **/
        private static double sigmoid(double z) {
            return (1 / (1 + Math.pow(Math.E,(-1 * z))));
        }

        /** TODO: Helper function for prediction **/
        /** Takes a test instance as input and outputs the probability of the label being 1 **/
        /** This function should call sigmoid() **/
        /** This does cross product of features and weights to create a value to plug into sigmoid **/
        /** to predict 1 or 0 (training data) **/
        private double probPred1(double[] x) {
           //no bias term (other lab)
           // from i -> d, summation(wi * Xi)
           //Classification Rule

           double probPredVal = 0.0;
           double crossProdVal = 0.0;
           for(int i = 0; i < weights.length; i++){
             crossProdVal += weights[i] * x[i];
           }
           probPredVal = sigmoid(crossProdVal);
           return probPredVal;
        }


        /** TODO: The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        /** This function should call probPred1() **/
        /** Takes testing data and predicts after the training data creates the weights. **/
        public int predict(double[] x) {

            double sum = probPred1(x);
            if(sum >= 0.5) return 1;
            else return 0;
        }

        /** This function takes a test set as input, call the predict() to predict a label for it, and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix **/
        public void printPerformance(List<LRInstance> testInstances) {
            double acc = 0;
            double p_pos = 0, r_pos = 0, f_pos = 0;
            double p_neg = 0, r_neg = 0, f_neg = 0;
            int TP=0, TN=0, FP=0, FN=0; // TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

            for(int i = 0; i < testInstances.size(); i++){
               double[] currInstance = testInstances.get(i).x;
               int realLabel = testInstances.get(i).label; //label taken from the instance (actual)
               //predict the label
               int prediction = predict(currInstance); //predicted label based off weights

               if(prediction == realLabel && realLabel == 1){
                 TP++;    // this is a true positive
               }
               else if(prediction == realLabel && realLabel == 0){
                 TN++;    // this is a true negative
               }
               else if(prediction != realLabel && realLabel == 1){
                 FN++;    //this is a False Negative
               }
               else if(prediction != realLabel && realLabel == 0){
                 FP++;    //this is a False Positive
               }
            }

            /**
            System.out.println("TP = "+TP);
            System.out.println("TN = "+TN);
            System.out.println("FN = "+FN);
            System.out.println("FP = "+FP);
            **/

            // Accuracy
            double num, den = 0.0;
            num = TP + TN;
            den = TP + TN + FP + FN;
            acc = num/den;

            //Precision positive
            num = TP;
            den = TP + FP;
            p_pos = num/den;

            //Precision Negative
            num = TN;
            den = TN + FN;
            p_neg = num/den;

            //Recall Positives
            num = TP;
            den = TP + FN;
            r_pos = num/den;

            //Recall Negative
            num = TN;
            den = TN + FP;
            r_neg = num/den;

            //F-Measure Positive
            f_pos = (2 * p_pos * r_pos) / (p_pos + r_pos);
            f_neg = (2 * p_neg  *r_neg) / (p_neg + r_neg);


            // TODO: write code here to compute the above mentioned variables

            System.out.println("Accuracy="+acc);
            System.out.println("P, R, and F1 score of the positive class=" + p_pos + " " + r_pos + " " + f_pos);
            System.out.println("P, R, and F1 score of the negative class=" + p_neg + " " + r_neg + " " + f_neg);
            System.out.println("Confusion Matrix");
            System.out.println(TP + "\t" + FN);
            System.out.println(FP + "\t" + TN);
        }


        /** Train the Logistic Regression using Stochastic Gradient Ascent **/
        /** Also compute the log-likelihood of the data in this function **/
        public void train(List<LRInstance> instances) {
            double lik = 0.0;
            for (int n = 0; n < ITERATIONS; n++) {
                lik = 0;
                for (int i = 0; i < instances.size(); i++) {
                    // TODO: Train the model
                    double[] instanceVal = instances.get(i).x; //X1, X2, X3...
                    double predictX = probPred1(instanceVal); //P(Y = 1 | X, W)
                    int labelVal = instances.get(i).label;

                    // learn the weight value based on prior weight at i
                    // Stochastic Gradient Ascent
                    // iterate through each feature?? of the instances

                    for(int j = 0; j < weights.length; j++){
                      double holder = 0.0;
                      weights[j] = (weights[j] + ((rate) * (labelVal-predictX) * instanceVal[j])) - (rate * lambda * weights[j]);
                      //double test = weights[j];
                      //System.out.println("Iteration = "+n+" Updated weight "+j+" at "+i+" = " +test);
                      // System.out.println(weights[j])
                    }

                    // TODO: Compute the log-likelihood of the data here. Remember to take logs when necessary

                    double holder2 = 0.0;
                    holder2 = labelVal * Math.log(probPred1(instanceVal)) + ((1-labelVal) * Math.log(1 - probPred1(instanceVal)));

                    lik += holder2;
                }
                System.out.println("iteration: " + n + " lik: " + lik);
            }
        }

        public static class LRInstance {
            public int label; // Label of the instance. Can be 0 or 1
            public double[] x; // The feature vector for the instance

            /** TODO: Constructor for initializing the Instance object **/
            public LRInstance(int label, double[] x) {
               this.label = label;
               this.x = x;
            }
        }

        /** Function to read the input dataset **/
        public static List<LRInstance> readDataSet(String file) throws FileNotFoundException {
            List<LRInstance> dataset = new ArrayList<LRInstance>();
            Scanner scanner = null;
            try {
                scanner = new Scanner(new File(file));

                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.startsWith("ju")) { // Ignore the header line
                        continue;
                    }
                    String[] columns = line.replace("\n", "").split(",");

                    // every line in the input file represents an instance-label pair
                    int i = 0;
                    double[] data = new double[columns.length - 1];
                    for (i=0; i < columns.length - 1; i++) {
                        data[i] = Double.valueOf(columns[i]);
                    }
                    int label = Integer.parseInt(columns[i]); // last column is the label
                    LRInstance instance = new LRInstance(label, data); // create the instance
                    dataset.add(instance); // add instance to the corpus
                }
            } finally {
                if (scanner != null)
                    scanner.close();
            }
            return dataset;
        }


        public static void main(String... args) throws FileNotFoundException {
            List<LRInstance> trainInstances = readDataSet("HW4_trainset.csv");
            List<LRInstance> testInstances = readDataSet("HW4_testset.csv");

            // create an instance of the classifier
            int d = trainInstances.get(0).x.length;
            LogisticRegression_withRegularization logistic = new LogisticRegression_withRegularization(d);

            logistic.train(trainInstances);

            System.out.println("Norm of the learned weights = "+logistic.weightsL2Norm());
            System.out.println("Length of the weight vector = "+logistic.weights.length);

            // printing accuracy for different values of lambda
            System.out.println("-----------------Printing train set performance-----------------");
            logistic.printPerformance(trainInstances);

            System.out.println("-----------------Printing test set performance-----------------");
            logistic.printPerformance(testInstances);
        }

    }
