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

    UPROPERTY(EditAnywhere)
    int32 NumInputs;

    UPROPERTY(EditAnywhere)
    int32 NumNeurons;

    UPROPERTY(EditAnywhere)
    TArray<float> Weights;

    UPROPERTY(EditAnywhere)
    TArray<float> Biases;
};

UCLASS()
class NEURALNET_API UNeuralNetworkData : public UDataAsset
{
	GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, Category = "Network")
    TArray<FLayerData> Layers;

    UPROPERTY(EditAnywhere, Category = "Network")
    float LearningRate = 0.1f;

    // Utilitaires pour transferer vers/depuis FNeuralNetwork
    void LoadFromNetwork(const FNeuralNetwork& Network);
    void SaveToNetwork(FNeuralNetwork& OutNetwork) const;
};
