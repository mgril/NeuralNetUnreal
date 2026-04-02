// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"

/**
 * 
 */
class NEURALNET_API NeuralLayer
{
public:
	NeuralLayer();
	~NeuralLayer();
};

struct FNeuralLayer
{
    int32 NumInputs;
    int32 NumNeurons;

    TArray<float> Weights;   // Taille : NumInputs * NumNeurons
    TArray<float> Biases;    // Taille : NumNeurons
    TArray<float> Outputs;   // Activations apres sigmoid

    // Valeurs intermediaires pour la backprop
    TArray<float> WeightedInputs; // Somme des inputs * weight +bias av sigmoid
    TArray<float> NodeValues; // dC/dWI stocke 

    void Initialize(int32 InNumInputs, int32 InNumNeurons);
    TArray<float> ForwardPass(const TArray<float>& Inputs);

    static float Sigmoid(float x);
    static float SigmoidDerivative(float x);

private:
    // Acces au poids entre neurone i (layer precedent) et neurone j (ce layer)
    // Weights[j * NumInputs + i]
    float GetWeight(int32 NeuronIndex, int32 InputIndex) const;

};