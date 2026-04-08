# Data Availability

## Planck CMB Data

All Planck satellite data used in this work are publicly available:

- **Planck Legacy Archive (PLA):** https://pla.esac.esa.int/
- **IRSA/IPAC Planck Archive:** https://irsa.ipac.caltech.edu/Missions/planck.html

The pipeline automatically downloads the required FFP10 simulation maps and SMICA CMB maps via `python run_pipeline.py --step download`.

## Simulated Training Data

Training data (simulated CMB skies with and without injected bubble collision signatures) are generated deterministically from fixed random seeds. To regenerate:

```bash
python run_pipeline.py --step generate --n-train 10000
```

All random seeds are documented in the code for full reproducibility.

## Trained Model Weights

Pre-trained ensemble weights are available upon request or via Zenodo:

- **Zenodo DOI:** [to be assigned upon publication]

Alternatively, retrain from scratch using `python run_pipeline.py --step train`.
