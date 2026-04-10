// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "NeuralLayer.h"

/**
 * 
 */
class NEURALNET_API FNeuralNetwork
{

public:
	FNeuralNetwork();
	~FNeuralNetwork();

public:
	TArray<FNeuralLayer> Layers;
	float LearningRate = 0.1f;
	void Build(const TArray<int32>& LayerSizes); // cree automatiquement les layers avec les bonnes dimensions
	TArray<float> Predict(const TArray<float>& Inputs);
	float ComputeCost(const TArray<float>& Output, const TArray<float>& Expected);
	float Train(const TArray<float>& Inputs, const TArray<float>& Expected);

private:
	
	void Backpropagate(const TArray<float>& Inputs, const TArray<float>& Expected);
};
