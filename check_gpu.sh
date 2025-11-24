#!/bin/bash

echo "========================================="
echo "  VÉRIFICATION ENVIRONNEMENT GPU"
echo "========================================="
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier carte NVIDIA
echo "1. Vérification carte NVIDIA..."
if lspci | grep -i nvidia > /dev/null; then
    GPU_NAME=$(lspci | grep -i nvidia | grep VGA | cut -d: -f3 | xargs)
    echo -e "${GREEN}✓ Carte détectée: $GPU_NAME${NC}"
else
    echo -e "${RED}✗ Aucune carte NVIDIA détectée${NC}"
    echo "  Ce projet nécessite une carte NVIDIA pour la version GPU"
    exit 1
fi

echo ""

# Vérifier nvidia-smi
echo "2. Vérification drivers NVIDIA..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    echo -e "${GREEN}✓ nvidia-smi trouvé (driver: $DRIVER_VERSION)${NC}"
    
    # Afficher infos GPU
    echo ""
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
else
    echo -e "${RED}✗ nvidia-smi non trouvé${NC}"
    echo "  Installez les drivers NVIDIA:"
    echo "  sudo apt install nvidia-driver-535"
    exit 1
fi

echo ""

# Vérifier CUDA
echo "3. Vérification CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | cut -d, -f1)
    echo -e "${GREEN}✓ CUDA Toolkit installé (version: $CUDA_VERSION)${NC}"
    echo "  Chemin nvcc: $(which nvcc)"
else
    echo -e "${YELLOW}⚠ CUDA Toolkit non trouvé${NC}"
    echo "  Installation recommandée:"
    echo "  sudo apt install nvidia-cuda-toolkit"
    echo ""
    echo "  Ou téléchargez depuis:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo ""

# Vérifier architecture GPU
echo "4. Vérification architecture GPU..."
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
ARCH_MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
ARCH_MINOR=$(echo $COMPUTE_CAP | cut -d. -f2)

echo "  Compute Capability: $COMPUTE_CAP"

if [ "$ARCH_MAJOR" -ge 6 ]; then
    echo -e "${GREEN}✓ Architecture supportée${NC}"
    
    # Recommander l'architecture SM
    SM_ARCH="sm_${ARCH_MAJOR}${ARCH_MINOR}"
    echo "  Architecture recommandée pour Makefile: $SM_ARCH"
else
    echo -e "${YELLOW}⚠ Architecture ancienne (< 6.0)${NC}"
    echo "  Les performances peuvent être limitées"
fi

echo ""

# Vérifier mémoire GPU
echo "5. Vérification mémoire GPU..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
GPU_MEM_GB=$(echo "scale=1; $GPU_MEM/1024" | bc)

if [ "$GPU_MEM" -ge 2048 ]; then
    echo -e "${GREEN}✓ Mémoire suffisante: ${GPU_MEM_GB} GB${NC}"
else
    echo -e "${YELLOW}⚠ Mémoire limitée: ${GPU_MEM_GB} GB${NC}"
    echo "  Minimum recommandé: 2 GB"
fi

# Vérifier mémoire disponible
GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
GPU_MEM_FREE_GB=$(echo "scale=1; $GPU_MEM_FREE/1024" | bc)
echo "  Mémoire disponible: ${GPU_MEM_FREE_GB} GB"

if [ "$GPU_MEM_FREE" -lt 512 ]; then
    echo -e "${YELLOW}⚠ Peu de mémoire disponible${NC}"
    echo "  Fermez les applications utilisant le GPU"
fi

echo ""

# Vérifier température
echo "6. Vérification température GPU..."
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
echo "  Température actuelle: ${GPU_TEMP}°C"

if [ "$GPU_TEMP" -lt 80 ]; then
    echo -e "${GREEN}✓ Température normale${NC}"
else
    echo -e "${YELLOW}⚠ Température élevée${NC}"
    echo "  Vérifiez le refroidissement"
fi

echo ""
echo "========================================="
echo "  RÉSUMÉ"
echo "========================================="
echo ""
echo -e "${GREEN}✓ Système prêt pour compilation GPU${NC}"
echo ""
echo "Prochaines étapes:"
echo "  1. Compiler: make gpu-cuda"
echo "  2. Tester: make test-gpu"
echo ""
echo "Performances attendues (estimation):"

# Estimer speedup basé sur GPU
case $GPU_NAME in
    *"RTX 40"*)
        echo "  Training: ~22x plus rapide que CPU"
        echo "  Inference: ~18x plus rapide que CPU"
        ;;
    *"RTX 30"*)
        echo "  Training: ~15x plus rapide que CPU"
        echo "  Inference: ~12x plus rapide que CPU"
        ;;
    *"RTX 20"*|*"GTX 16"*)
        echo "  Training: ~12x plus rapide que CPU"
        echo "  Inference: ~10x plus rapide que CPU"
        ;;
    *"GTX 10"*)
        echo "  Training: ~10x plus rapide que CPU"
        echo "  Inference: ~8x plus rapide que CPU"
        ;;
    *)
        echo "  Training: ~8-15x plus rapide que CPU"
        echo "  Inference: ~6-12x plus rapide que CPU"
        ;;
esac

echo ""
echo "========================================="
