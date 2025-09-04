import torch
import sys

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from simple_knn import knn
    print("✅ DreamGaussian's CUDA extensions imported successfully.")

    if torch.cuda.is_available():
        print("✅ CUDA is available. Running a basic tensor operation.")
        device = torch.device("cuda")
        x = torch.randn(5, 5).to(device)
        print("Test tensor on GPU:", x.device)
        
        # A simple check to ensure the module is callable without a full pipeline
        _ = knn(torch.randn(1, 3).to(device), torch.randn(10, 3).to(device))
        print("✅ simple-knn module is functional.")
    else:
        print("❌ CUDA not available. DreamGaussian will not work without a GPU.")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Failed to import DreamGaussian modules. Error: {e}")
    print("Please check your installation and ensure the CUDA compiler was successful.")
    sys.exit(1)

except Exception as e:
    print(f"❌ An unexpected error occurred during validation: {e}")
    sys.exit(1)
