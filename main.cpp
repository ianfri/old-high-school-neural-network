// g++ -std=c++11 -o main_obv.o main_more_obvious_pattern.cc

// Copyright Ian Friedrichs, 2018

/*

L1   L2 OP
x    x

x    x

x    x
          x
x    x

x    x

x    x

*/

#include <iostream>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <thread>

#define LEARNING_RATE 0.01
#define ITERATIONS 10

using namespace std;

/* GLOBALS FOR USE IN LOOPS AND COUNTING */
int currentLayer = 0;
int currentDataRow = 0;
int currentDataCol = 0;
float currentWeightedSum;



/* INTERNAL DATA STRUCTURES */

// define a node
struct node {
  double value;
  double weights[6]; // 6 weights stemming from each node
  double incoming_weight_sum;
  double error_signal;
};

// define a layer of nodes
struct layer {
  struct node Nodes[6];
};

// initiate two layers of 6 nodes
struct layer Layers[2];

// initiate the output node
struct node oNode;

// difference between actual output and expected output
double delta_k;

// error signal for output node
double error_k;

// sum of the weights between L2 and output node each multiplied by the error signal of the output node
double sigma_weights_L2_Outp_errork;

// error signal for L2's nodes
double error_L2;


/* GIVEN DATA for training */
vector<vector<int> > givenInputs { // randomly generated on calculator
{ 0,0,0,1,0,1 }, // 0
{ 1,1,0,1,0,1 }, // 1
{ 0,0,1,0,1,1 }, // 0
{ 1,1,0,1,1,1 }, // 1
{ 0,1,0,0,0,1 }, // 1
{ 0,1,1,0,1,1 }, // 1
{ 0,0,0,1,0,0 }, // 0
{ 0,0,1,1,1,1 }, // 0
{ 0,0,0,0,0,1 }, // 0
{ 1,1,0,1,0,1 }, // 1
};

vector<int> givenOutputs { 0,1,0,1,1,1,0,0,0,1 };

/* FUNCTIONS */

// actual sigmoid function
double my_sigmoid(double iws) {
  return (double)(1 / (1 + pow(2.71828182, (-1)*iws)));
}

// reset the incoming weight sums for the nodes in L2 (hidden layer)
void reset_incoming_weight_sum_L2() {
  for (int i=0; i < 6; i++) { // for every node in L2
    Layers[1].Nodes[i].incoming_weight_sum = 0;
  }
}

// reset the incoming weight sum for oNode
void reset_incoming_weight_sum() {
  oNode.incoming_weight_sum = 0;
}

// adjust the weights from Layer 1 to Layer 2
void adjustWeights_L1toL2() {
  for (int j=0; j < 6; j++) { // for each node in L2
    for (int i=0; i < 6; i++) { // for each node in L1
      Layers[0].Nodes[i].weights[j] = Layers[0].Nodes[i].weights[j] + (LEARNING_RATE * error_L2 * Layers[1].Nodes[j].value);
      cout << "Layers[0].Nodes[" << i << "].weights[" << j << "] = Layers[0].Nodes[" << i << "].weights[" << j << "] + (LR {" << LEARNING_RATE << "} * error_L2 {" << error_L2 << "} * Layers[1].Nodes[" << j << "].value {" << Layers[1].Nodes[j].value << "})" << endl;
    }
  }
}

// calculate the error signal for L2's nodes
void calculateErrorSignal_L2() {
  error_L2 = (delta_k * oNode.value * sigma_weights_L2_Outp_errork);
  cout << "error_L2 = delta_k {" << delta_k << "} * oNode.value {" << oNode.value << "} * sigma_weights_L2_Outp_errork {" << sigma_weights_L2_Outp_errork << "}" << endl;
  cout << "error_L2 = " << error_L2 << endl;
}

// for use in calculateErrorSignal_L2: the sum of the weights between L2 and output node each multiplied by the error signal of the output node
void calculate_signma_weights_L2_Outp_errork() { 
  for (int i=0; i < 6; i++) {
    sigma_weights_L2_Outp_errork += (Layers[1].Nodes[i].weights[0] * error_k);
    cout << "sigma_weights_L2_Outp_errork += " << "Layers[1].Nodes[ " << i << "].weights[0] * " << "error_k {" << error_k << "}" << endl;
  } 
  cout << "sigma_weights_L2_Outp_errork = " << sigma_weights_L2_Outp_errork << endl;
}

// adjust the weights for weights between L2 and the output node
void adjustWeights_L2toOutput() {
  double adjustedWeight;
  for (int i=0; i < 6; i++) { // for every node in L2
    for (int j=0; j < 6; j++) { // for every weight in each node
      cout << "unadjusted weight = " << Layers[1].Nodes[i].weights[0] << endl;
      adjustedWeight = Layers[1].Nodes[i].weights[0] + (LEARNING_RATE * error_k * oNode.value); // last term previously Layers[1].Nodes[i].value, this was incorrect
Layers[1].Nodes[i].weights[0] = adjustedWeight;
      cout << "adjustedWeight = " << adjustedWeight << endl;
      //cout << "Layers[1].Nodes[" << i << "].weights[" << j << "] = " << Layers[1].Nodes[i].weights[j] << " + (" << LEARNING_RATE << " * " << error_k << " * " << Layers[1].Nodes[i].value << ")" << endl;
      cout << "    so Layers[1].Nodes[" << i << "].weights[" << j << "] = " << Layers[1].Nodes[i].weights[j] << endl;
    }
  }
}

// calculate the error signals for each node in Layer 2
void calculateErrorSignal() {
  delta_k = givenOutputs[currentDataRow] - oNode.value;
  cout << "delta_k = givenOutputs[" << currentDataRow << "] - " << oNode.value << ", which = " << delta_k << endl;
  error_k = delta_k * oNode.value * (1 - oNode.value);
  cout << "error_k = " << delta_k << " * " << oNode.value << " * " << (1-oNode.value) << endl;
}

// calculate activation function for the output
void calcActivationFunction_forOutput() {
  oNode.value = my_sigmoid(oNode.incoming_weight_sum);
  cout << "oNode.value = " << oNode.value << endl;
}

// calculate weighted sums for output node
void calcWeightedSums_forOutput() {
  for (int i=0; i < 6; i++) { // for each node in L2
    oNode.incoming_weight_sum += (Layers[1].Nodes[i].weights[0] * Layers[1].Nodes[i].value); // each of L2's nodes only has one weight (weights[0]) going to the singular output node
    cout << "adding: " << Layers[1].Nodes[i].weights[0] << " * " << Layers[1].Nodes[i].value << ", which = " << Layers[1].Nodes[i].weights[0] * Layers[1].Nodes[i].value << endl;
    cout << "oNode.incoming_weight_sum = " << oNode.incoming_weight_sum << endl;
  }
}

// execute sigmoid activation function on each node
void calcActivationFunction() {
  for (int i=0; i < 6; i++) { // for each node in L2
    Layers[1].Nodes[i].value = my_sigmoid(Layers[1].Nodes[i].incoming_weight_sum);
    cout << "L2 node " << i << " = my_sig(" << Layers[1].Nodes[i].incoming_weight_sum << ") = " << Layers[1].Nodes[i].value << endl;
  }
}

// calculate weighted sums
void calcWeightedSums() {
  for (int i=0; i < 6; i++) { // for each node in L2
    for (int j=0; j < 6; j++) { // for each node in L1
      Layers[1].Nodes[i].incoming_weight_sum += (Layers[0].Nodes[j].weights[i] * Layers[0].Nodes[j].value);
      cout << "adding: " << Layers[0].Nodes[j].weights[i] << " * " << Layers[0].Nodes[j].value << endl;
      cout << "Layers[1].Nodes[" << i << "].incoming_weight_sum = " << Layers[1].Nodes[i].incoming_weight_sum << endl;
    }
  }
}

// randomly initialize all weights
void initialize_weights() {
  double rndm;
  for (int i=0; i < 6; i++) { // iterates through nodes in Layer 1
    for (int j=0; j < 6; j++) { // iterates through weights of each node
      rndm = rand()/(double)RAND_MAX;
      Layers[0].Nodes[i].weights[j] = rndm;
      cout << "Layers[" << 0 << "].Nodes[" << i << "].weights[" << j << "] = " << rndm << endl;
    }
  }
  for (int k=0; k < 6; k++) { // iterates through nodes in Layer 2
    for (int n=0; n < 6; n++) { // iterates through weights of each node
      rndm = rand()/(double)RAND_MAX;
      Layers[1].Nodes[k].weights[n] = rndm;
      cout << "Layers[1].Nodes[" << k << "].weights[" << n << "] = " << rndm << endl;
    }
  }
}

// feed one row of input data into the first layer
void present_pattern() {
  int i;
  for (i=0; i < 6; i++) { // iterates through nodes in layer 1 (input)
    Layers[0].Nodes[i].value = givenInputs[currentDataRow][i];
    cout << "present_pattern: " << Layers[0].Nodes[i].value << endl;
  }
}

/* ##############################################################################################################
  ##################### START OF FUNCTION DEFINITIONS FOR TESTING THE NN #######################################
  ############################################################################################################## */
// given testing data
vector<vector<int> > testingInputs { // randomly generated on calculator
  { 0,1,1,0,0,0 }, // 1
  { 0,0,1,0,0,0 }, // 0
  { 0,1,1,1,0,1 }, // 1
  { 0,0,1,1,1,0 }, // 0
  { 1,1,1,1,0,1 }, // 1
};
vector<int> testingOutputs { 1,0,1,0,1 };

double testingDataRow = 0;

struct testinganswers {
int right;
int wrong;
};
struct testinganswers testing_Answers;

// feed one row of testing input data to the first layer
void testing_present_pattern() {
  int i;
  for (i=0; i < 6; i++) { // for every node in L1
    Layers[0].Nodes[i].value = testingInputs[testingDataRow][i];
    cout << "training_present_pattern: " << Layers[0].Nodes[i].value;
  }
}

// calculate the weighted sums
void testing_calcWeightedSums() {
  for (int i=0; i < 6; i++) { // for each node in L2
    for (int j=0; j < 6; j++) { // for each node in L1
      Layers[1].Nodes[i].incoming_weight_sum += (Layers[0].Nodes[j].weights[i] * Layers[0].Nodes[j].value);
      cout << "adding: " << Layers[0].Nodes[j].weights[i] << " * " << Layers[0].Nodes[j].value << endl;
      cout << "Layers[1].Nodes[" << i << "].incoming_weight_sum = " << Layers[1].Nodes[i].incoming_weight_sum << endl;
    }
  }
}

// execute sigmoid activation function on each node
void testing_calcActivationFunction() {
  for (int i=0; i < 6; i++) { // for each node in L2
    Layers[1].Nodes[i].value = my_sigmoid(Layers[1].Nodes[i].incoming_weight_sum);
    cout << "L2 node " << i << " = my_sig(" << Layers[1].Nodes[i].incoming_weight_sum << ") = " << Layers[1].Nodes[i].value << endl;
  }
}

// calculate weighted sums for output node
void testing_calcWeightedSums_forOutput() {
  for (int i=0; i < 6; i++) { // for each node in L2
    oNode.incoming_weight_sum += (Layers[1].Nodes[i].weights[0] * Layers[1].Nodes[i].value); // each of L2's nodes only has one weight (weights[0]) going to the singular output node
    cout << "adding: " << Layers[1].Nodes[i].weights[0] << " * " << Layers[1].Nodes[i].value << ", which = " << Layers[1].Nodes[i].weights[0] * Layers[1].Nodes[i].value << endl;
    cout << "oNode.incoming_weight_sum = " << oNode.incoming_weight_sum << endl;
  }
}

// calculate activation function for the output
void testing_calcActivationFunction_forOutput() {
  oNode.value = my_sigmoid(oNode.incoming_weight_sum);
  cout << "oNode.value = " << oNode.value << endl;
}

void testing_print_whether_right() {
  double ans = testingOutputs[testingDataRow] - oNode.value;
  if (ans < 0.5) {
    cout << "%%%%%%%% correct! %%%%%%%%" << endl;
    testing_Answers.right += 1;
  } else {
    cout << "%%%%%%% incorrect! %%%%%%%" << endl;
    testing_Answers.wrong += 1;
  }
}


/* ##############################################################################################################
  ##################### END OF FUNCTION DEFINITIONS FOR TESTING THE NN #########################################
  ############################################################################################################## */


int main() {
// set random seed based on time
time_t now = time(0);
srand(now);
// initialize the random weights
initialize_weights();
// main loop
for (int mn = 0; mn < ITERATIONS; mn++) { // repeat the training 100 times
  int j;
  for (j=0; j < 10; j++) { // for every pattern in the training set
    present_pattern();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    calcWeightedSums(); 
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    calcActivationFunction();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    calcWeightedSums_forOutput();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    calcActivationFunction_forOutput();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    calculateErrorSignal();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    adjustWeights_L2toOutput();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    calculate_signma_weights_L2_Outp_errork();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    calculateErrorSignal_L2();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    adjustWeights_L1toL2();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    reset_incoming_weight_sum();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    reset_incoming_weight_sum_L2();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // prepare for next data row
    currentDataRow++;
  }
  currentDataRow = 0; // reset the current data row used in present_pattern()
}
// print out the final weights
cout << "FINAL WEIGHTS: " << endl << endl;
for (int n=0; n < 6; n++) { // for each node in L1
  for (int b=0; b < 6; b++) { // for each weight from each node in L1
    cout << "Layers[0].Nodes[" << n << "].weights[" << b << "] = " << Layers[0].Nodes[n].weights[b] << endl;
  }
}
for (int n=0; n < 6; n++) { // for each node in L2
  cout << "Layers[1].Nodes[" << n << "].weights[0] = " << Layers[1].Nodes[n].weights[0] << endl;
}
/* testing the newly minted NN */
cout << "TESTING NEW NN" << endl << endl << endl;
for (int nn = 0; nn < 5; nn++) { // for each row of data in the testing set
  testing_present_pattern();
  testing_calcWeightedSums();
  testing_calcActivationFunction();
  testing_calcWeightedSums_forOutput();
  testing_calcActivationFunction_forOutput();
  testing_print_whether_right();
  reset_incoming_weight_sum();
  reset_incoming_weight_sum_L2();
  // prepare for next data row
  testingDataRow++;
}
cout << "right: " << testing_Answers.right << endl;
cout << "wrong: " << testing_Answers.wrong << endl;
return 0;
}
