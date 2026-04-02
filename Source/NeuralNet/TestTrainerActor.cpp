// Fill out your copyright notice in the Description page of Project Settings.
#include "TestTrainerActor.h"
#include "NeuralNetwork.h"


// Sets default values
ATestTrainerActor::ATestTrainerActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ATestTrainerActor::BeginPlay()
{
	Super::BeginPlay();
    // Architecture : 3 inputs ? 4 neurones cachés ? 2 outputs
    FNeuralNetwork Network;
    Network.Build({ 3, 4, 2 });

    // Inputs de test
    TArray<float> Inputs = { 0.5f, 0.8f, 0.2f };
    TArray<float> Expected = { 1.0f, 0.0f };

    // Predict chaine les deux layers automatiquement
    TArray<float> Outputs = Network.Predict(Inputs);

    // Log des outputs finaux
    for (int32 i = 0; i < Outputs.Num(); i++)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("Output [%d]  %.4f  (Expected: %.4f)"),
            i,
            Outputs[i],
            Expected[i]
        );
    }

    // Log du cout global
    float Cost = Network.ComputeCost(Outputs, Expected);
    UE_LOG(LogTemp, Warning, TEXT("Cost  %.4f"), Cost);

    // Verification de l'architecture
    UE_LOG(LogTemp, Warning,
        TEXT("Network: %d layers  Layer[0]: %d %d  Layer[1]: %d %d"),
        Network.Layers.Num(),
        Network.Layers[0].NumInputs, Network.Layers[0].NumNeurons,
        Network.Layers[1].NumInputs, Network.Layers[1].NumNeurons
    );
}

// Called every frame
void ATestTrainerActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

