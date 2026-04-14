#include "TestTrainerActor.h"
#include "NeuralNetworkData.h"

#if WITH_EDITOR
#include "AssetRegistry/AssetRegistryModule.h"
#include "UObject/SavePackage.h"
#include "Misc/PackageName.h"
#endif
#include "ImagePixelUtils.h"

ATestTrainerActor::ATestTrainerActor()
{
    PrimaryActorTick.bCanEverTick = false;
}

void ATestTrainerActor::BeginPlay()
{
    Super::BeginPlay();


    TArray<TArray<float>> InputsImg;
    TArray<TArray<float>> Labels;

    // Charge le dataset  10 classes, images redimensionnees en 28x28
    FImagePixelUtils::LoadDataset(
        TEXT("/Game/testSet/LabelizedImg"),
        InputsImg,
        Labels,
        10,   //  10 classes (0 à 9)
        28    //  28x28 pixels
    );

    // Vérifie que les pixels sont bien entre 0 et 1
    if (InputsImg.Num() > 0)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("Premiere image : %d pixels"), InputsImg[0].Num());
        UE_LOG(LogTemp, Warning,
            TEXT("Pixel[0]=%.3f  Pixel[100]=%.3f  Pixel[400]=%.3f"),
            InputsImg[0][0], InputsImg[0][100], InputsImg[0][400]);
        UE_LOG(LogTemp, Warning,
            TEXT("Label image 0 : [%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f]"),
            Labels[0][0], Labels[0][1], Labels[0][2], Labels[0][3], Labels[0][4],
            Labels[0][5], Labels[0][6], Labels[0][7], Labels[0][8], Labels[0][9]);
    }

    FNeuralNetwork Network;

    // CAS 1 : un asset est assigne dans l'editeur  on charge
    if (NetworkAsset && NetworkAsset->Layers.Num() > 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("=== Chargement depuis l'asset ==="));
        NetworkAsset->LoadIntoNetwork(Network);
    }
    // CAS 2 : pas d'asset  on entraîne puis on sauvegarde 
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("=== Entraînement from scratch ==="));

        Network.Build({ 2, 4, 1 });
        Network.LearningRate = 0.5f;

        TArray<TArray<float>> Inputs = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
        TArray<TArray<float>> Expected = { { 0 },  { 1 },  { 1 },  { 0 } };

        for (int32 Epoch = 0; Epoch < 10000; Epoch++)
            for (int32 i = 0; i < Inputs.Num(); i++)
                Network.Train(Inputs[i], Expected[i]);

        UE_LOG(LogTemp, Warning, TEXT("Entraînement termine !"));

#if WITH_EDITOR
        // Creer et sauvegarder l'asset sur disque
        FString AssetName = TEXT("NN_XOR");
        FString PackagePath = FString::Printf(TEXT("/Game/%s"), *AssetName);

        UPackage* Package = CreatePackage(*PackagePath);
        Package->FullyLoad();

        UNeuralNetworkData* Asset = NewObject<UNeuralNetworkData>(
            Package,
            UNeuralNetworkData::StaticClass(),
            *AssetName,
            RF_Public | RF_Standalone | RF_Transactional
        );

        Asset->SaveFromNetwork(Network);
        FAssetRegistryModule::AssetCreated(Asset);
        Package->MarkPackageDirty();

        FSavePackageArgs SaveArgs;
        SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;

        FString FilePath = FPackageName::LongPackageNameToFilename(
            PackagePath,
            FPackageName::GetAssetPackageExtension()
        );

        bool bSaved = UPackage::SavePackage(Package, Asset, *FilePath, SaveArgs);
        UE_LOG(LogTemp, Warning, TEXT("Asset sauvegarde : %s"), bSaved ? TEXT("OUI") : TEXT("NON"));
#endif
    }

    // Prediction finale dans les deux cas 
    TArray<float> R00 = Network.Predict({ 0.0f, 0.0f });
    TArray<float> R01 = Network.Predict({ 0.0f, 1.0f });
    TArray<float> R10 = Network.Predict({ 1.0f, 0.0f });
    TArray<float> R11 = Network.Predict({ 1.0f, 1.0f });

    UE_LOG(LogTemp, Warning, TEXT("0 XOR 0 = %.4f (attendu: 0)"), R00[0]);
    UE_LOG(LogTemp, Warning, TEXT("0 XOR 1 = %.4f (attendu: 1)"), R01[0]);
    UE_LOG(LogTemp, Warning, TEXT("1 XOR 0 = %.4f (attendu: 1)"), R10[0]);
    UE_LOG(LogTemp, Warning, TEXT("1 XOR 1 = %.4f (attendu: 0)"), R11[0]);
}

