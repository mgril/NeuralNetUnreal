// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "NeuralNetwork.h"
#include "Engine/DataAsset.h"
#include "NeuralNetworkData.generated.h"

/**
 * 
 */

USTRUCT(BlueprintType)
struct FLayerData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Layer")
    int32 NumInputs = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Layer")
    int32 NumNeurons = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Layer")
    TArray<float> Weights;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Layer")
    TArray<float> Biases;
};

UCLASS()
class NEURALNET_API UNeuralNetworkData : public UPrimaryDataAsset
{
	GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, Category = "Network")
    TArray<FLayerData> Layers;

    UPROPERTY(EditAnywhere, Category = "Network")
    float LearningRate = 0.1f;


    void SaveFromNetwork(const FNeuralNetwork& Network);

    bool LoadIntoNetwork(FNeuralNetwork& OutNetwork) const;
};
