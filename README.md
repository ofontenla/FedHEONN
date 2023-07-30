# FedHEONN

Federated and homomorphically encrypted learning method for one-layer neural networks.

FedHEONN is a Federated Learning (FL) method based on a neural network without hidden layers that incorporates homomorphic encryption (HE). Unlike traditional FL methods that require multiple rounds of training for convergence, our method obtains the collaborative global model in a single training round, yielding an effective and efficient model that simplifies management of the FL training process. In addition, since our method includes HE, it is also robust against model inversion attacks.

## Prerequisites

The implementation employs the following libraries:

- [tenseal](https://github.com/OpenMined/TenSEAL#features)
- numpy
- scipy

To run the usage examples ([example_classification.py](https://github.com/ofontenla/FedHEONN/blob/main/src/example_classification.py) and [example_regression.py](https://github.com/ofontenla/FedHEONN/blob/main/src/example_regression.py)) the following libraries are also needed:

- pandas
- sklearn


## Publications

This method was published in the following article:

O. Fontenla-Romero, B. Guijarro-Berdiñas, E. Hernández-Pereira, B. Pérez-Sánchez (2023) [FedHEONN: Federated and homomorphically encrypted learning method for one-layer neural networks](https://doi.org/10.1016/j.future.2023.07.018). Future Generation Computer Systems. In Press, Journal Pre-proof. Open Access.

    @article{fedHEONN2023,
        author = {Oscar Fontenla-Romero and Bertha Guijarro-Berdiñas and Elena Hernández-Pereira and Beatriz Pérez-Sánchez},
        title = {FedHEONN: Federated and homomorphically encrypted learning method for one-layer neural networks}, 
        journal = {Future Generation Computer Systems},
        volume = {149},
        pages  = {200-211},
        year = {2023},
        doi = {https://doi.org/10.1016/j.future.2023.07.018},
    }

## License

[GPL-3.0-only](https://opensource.org/license/gpl-3-0/)
