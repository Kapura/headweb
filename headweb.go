package main

import "fmt"
import "math"
import "math/rand"
import "time"

type NeuralNet interface {
	Calculate([]float64) []float64 // calculate output layer given an input layer
	Train([]float64, []float64)    // given an input and output layer, train the network on that case
	SetLearningRate(float64)       // adjust the learning rate of the network
	Activate(float64) float64      // activation function for the neuron (not implemented in SN)
}

// NN using linear neurons
type SimpleNet struct {
	inputLayer []float64 // input neurons

	i_h_weights  [][]float64 // input-hidden connections
	i_h_err_grad [][]float64

	hiddenLayer []float64 // hidden neurons

	h_o_weights  [][]float64 // hidden-output connections
	h_o_err_grad [][]float64

	outputLayer []float64 // output neurons

	learningRate float64
}

func NewSimpleNet(i, h, o int) *SimpleNet {
	sn := &SimpleNet{
		inputLayer:   make([]float64, i),
		i_h_weights:  make([][]float64, i),
		i_h_err_grad: make([][]float64, i),
		hiddenLayer:  make([]float64, h),
		h_o_weights:  make([][]float64, h),
		h_o_err_grad: make([][]float64, h),
		outputLayer:  make([]float64, o),
		learningRate: 1,
	}

	for s := range sn.i_h_weights {
		sn.i_h_weights[s] = make([]float64, h)
		sn.i_h_err_grad[s] = make([]float64, h)
	}

	for s := range sn.h_o_weights {
		sn.h_o_weights[s] = make([]float64, o)
		sn.h_o_err_grad[s] = make([]float64, o)
	}

	return sn
}

func (sn *SimpleNet) Calculate(inputs []float64) []float64 {
	if len(inputs) != len(sn.inputLayer) {
		panic("Input size mismatch!")
	}

	copy(sn.inputLayer, inputs)
	sn.forwardPropogate()

	return sn.outputLayer
}

func (sn *SimpleNet) Train(inputs, targets []float64) {
	if len(inputs) != len(sn.inputLayer) {
		panic("Input size mismatch!")
	}
	if len(targets) != len(sn.outputLayer) {
		panic("Target size mismatch!")
	}

	copy(sn.inputLayer, inputs)
	sn.forwardPropogate()

	sn.backPropogate(targets)
}

func (sn *SimpleNet) SetLearningRate(r float64) {
	sn.learningRate = r
}

func (sn *SimpleNet) LogisticActivate(v float64) float64 {
	// logistic neuron
	return (1 / (1 + math.Exp(-v)))
}

func (sn *SimpleNet) forwardPropogate() {
	var sum float64
	for h_idx := range sn.hiddenLayer {
		sum = 0
		for i_idx := range sn.inputLayer {
			sum += sn.inputLayer[i_idx] * sn.i_h_weights[i_idx][h_idx]
		}

		sn.hiddenLayer[h_idx] = sn.LogisticActivate(sum)
	}

	for o_idx := range sn.outputLayer {
		sum = 0
		for h_idx := range sn.hiddenLayer {
			sum += sn.hiddenLayer[h_idx] * sn.h_o_weights[h_idx][o_idx]
		}

		sn.outputLayer[o_idx] = sn.LogisticActivate(sum)
	}
}

func (sn *SimpleNet) backPropogate(targets []float64) {
	var activation, residual_err, in_val float64

	// compute error grads on output layer
	for h_idx := range sn.h_o_err_grad {
		in_val = sn.hiddenLayer[h_idx]
		for o_idx := range sn.h_o_err_grad[h_idx] {
			activation = sn.outputLayer[o_idx]
			residual_err = sn.outputLayer[o_idx] - targets[o_idx]
			sn.h_o_err_grad[h_idx][o_idx] = in_val * residual_err * activation * (1 - activation)
		}
	}

	// compute error grads on hidden layer
	for i_idx := range sn.i_h_err_grad {
		in_val = sn.inputLayer[i_idx]
		for h_idx := range sn.i_h_err_grad[i_idx] {
			activation = sn.hiddenLayer[h_idx]
			residual_err = 0
			// sum of grads / inputs * weights
			for o_idx, g := range sn.h_o_err_grad[h_idx] {
				residual_err += (g / sn.hiddenLayer[h_idx]) * sn.h_o_weights[h_idx][o_idx]
			}
			sn.i_h_err_grad[i_idx][h_idx] = in_val * residual_err * activation * (1 - activation)
		}
	}

	//update weights
	for i_idx := range sn.i_h_weights {
		for h_idx := range sn.i_h_weights[i_idx] {
			sn.i_h_weights[i_idx][h_idx] -= sn.learningRate * sn.i_h_err_grad[i_idx][h_idx]
		}
	}

	for h_idx := range sn.h_o_weights {
		for o_idx := range sn.h_o_weights[h_idx] {
			sn.h_o_weights[h_idx][o_idx] -= sn.learningRate * sn.h_o_err_grad[h_idx][o_idx]
		}
	}
}

func (sn *SimpleNet) randomiseWeights() {
	for i_idx := range sn.i_h_weights {
		for h_idx := range sn.i_h_weights[i_idx] {
			sn.i_h_weights[i_idx][h_idx] = (rand.Float64() - .5)
		}
	}

	for h_idx := range sn.h_o_weights {
		for o_idx := range sn.h_o_weights[h_idx] {
			sn.h_o_weights[h_idx][o_idx] = (rand.Float64() - .5)
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	nn := NewSimpleNet(5, 3, 3) // five input neurons, three hidden neurons, three output neurons
	nn.randomiseWeights()
	nn.SetLearningRate(.2)

	for i := 0; i < 3000; i++ {
		var trainingData = make([]float64, 5)
		var targetData = make([]float64, 3)
		var binSum float64 = 0

		for d := range trainingData {
			if rand.Float64() > .5 {
				trainingData[d] = 1
			} else {
				trainingData[d] = 0
			}
			binSum += math.Pow(2, float64(4-d)) * trainingData[d]
		}

		if binSum > 10 {
			targetData[0] = 1
		} else {
			targetData[0] = 0
		}

		if binSum > 5 {
			targetData[1] = 1
		} else {
			targetData[1] = 0
		}

		if binSum > 3 {
			targetData[2] = 1
		} else {
			targetData[2] = 0
		}

		nn.Train(trainingData, targetData)
	}

	fmt.Println(nn.Calculate([]float64{1, 0, 1, 1, 0}))
	fmt.Println(nn.Calculate([]float64{0, 0, 0, 1, 0}))
	fmt.Println(nn.Calculate([]float64{0, 0, 1, 0, 0}))
	fmt.Println(nn.Calculate([]float64{0, 1, 0, 0, 0}))

}
