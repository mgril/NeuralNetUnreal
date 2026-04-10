// Fill out your copyright notice in the Description page of Project Settings.
#include "TestTrainerActor.h"
#include "NeuralNetwork.h"

#if WITH_EDITOR
#include "AssetRegistry/AssetRegistryModule.h"
#include "UObject/SavePackage.h"
#endif

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

#if WITH_EDITOR
    
    FString PackagePath = TEXT("/Game/NeuralNetworks/NN_XOR");
    UPackage* Package = CreatePackage(*PackagePath);

    UNeuralNetworkData* NewAsset = NewObject<UNeuralNetworkData>(
        Package,
        UNeuralNetworkData::StaticClass(),
        TEXT("NN_XOR"),
        RF_Public | RF_Standalone
    );

    FNeuralNetwork Network;
    Network.Build({ 2, 4, 1 });
    Network.LearningRate = 0.5f;

    TArray<TArray<float>> Inputs = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
    TArray<TArray<float>> Expected = { { 0 },  { 1 },  { 1 },  { 0 } };

    for (int32 Epoch = 0; Epoch < 10000; Epoch++)
        for (int32 i = 0; i < Inputs.Num(); i++)
            Network.Train(Inputs[i], Expected[i]);

    NewAsset->SaveFromNetwork(Network);

    FSavePackageArgs SaveArgs;
    SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
    FString FilePath = FPackageName::LongPackageNameToFilename(
        PackagePath,
        FPackageName::GetAssetPackageExtension()
    );
    UPackage::SavePackage(Package, NewAsset, *FilePath, SaveArgs);

    UE_LOG(LogTemp, Warning, TEXT("Asset créé : %s"), *FilePath);
#endif

    // charger depuis l'asset si il existe deja
    if (NetworkAsset && NetworkAsset->Layers.Num() > 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Chargement depuis l'asset..."));
        NetworkAsset->LoadIntoNetwork(Network);
    }
    else
    {
        // from scratch
        UE_LOG(LogTemp, Warning, TEXT("Entrainement from scratch..."));

        Network.Build({ 2, 4, 1 });
        Network.LearningRate = 0.5f;

        TArray<TArray<float>> TrainingInputs = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
        TArray<TArray<float>> TrainingExpected = { { 0 },  { 1 },  { 1 },  { 0 } };

        for (int32 Epoch = 0; Epoch < 10000; Epoch++)
        {
            for (int32 i = 0; i < TrainingInputs.Num(); i++)
                Network.Train(TrainingInputs[i], TrainingExpected[i]);
        }

        // Sauvegarder dans l'asset
        if (NetworkAsset)
        {
            NetworkAsset->SaveFromNetwork(Network);
            UE_LOG(LogTemp, Warning, TEXT("Poids sauvegardes dans l'asset !"));
        }
    }

    // Prediction finale
    TArray<float> Result = Network.Predict({ 1.0f, 0.0f });
    UE_LOG(LogTemp, Warning, TEXT("1 XOR 0  %.4f"), Result[0]);
}

void ATestTrainerActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

